"""Read-only pilot acquisition through authorized SSH and the legacy mongo shell."""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time

from chemtellect.data_integrity import validate_jsonl

METHOD = "mongodb_sample_first_stage"
FIELDS = ["_id", "code", "smiles", "inchikey", "formula"]
DIAGNOSTIC_LIMIT = 64 * 1024
SOURCE_ID_ENCODING = (
    'bson-id-v1: objectid:<lowercase hex> or string:<JSON string literal>; '
    'other BSON types rejected'
)
DEFAULT_TRANSFER_LIMIT = 10_000  # records, not a minimum or a byte-size estimate
LIMITATIONS = (
    "MongoDB $sample first stage may use the optimized pseudo-random cursor when "
    "the collection has over 100 documents and the requested size is below 5% "
    "of collection size. Actual planner behavior is unverified (no explain obtained). "
    "Selection bias is possible; this is not a guaranteed uniform or representative "
    "sample and does not support unbiased population estimates. Independent packages "
    "may overlap and do not constitute a frozen snapshot."
)


@dataclass(frozen=True)
class SamplingConfig:
    ssh_target: str
    database: str
    collection: str
    package: str
    size: int = 1200
    server_timeout_ms: int = 30_000
    client_timeout: float = 120
    allow_large_sample: bool = False
    raw_root: Path = Path("data/raw")
    max_transfer_bytes: int = 100 * 1024 * 1024

    def validate(self) -> None:
        for name in ("database", "collection", "package"):
            value = getattr(self, name)
            safe_name = r"[A-Za-z0-9_][A-Za-z0-9_-]{0,127}"
            if not isinstance(value, str) or not re.fullmatch(safe_name, value):
                raise ValueError(f"Unsafe {name}")
        if not isinstance(self.ssh_target, str) or not re.fullmatch(
            r"(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9_][A-Za-z0-9_.-]*", self.ssh_target
        ):
            raise ValueError("Unsafe SSH target (use a configured host alias)")
        for name in ("size", "server_timeout_ms", "max_transfer_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.client_timeout, bool)
            or not math.isfinite(self.client_timeout)
            or self.client_timeout <= 0
        ):
            raise ValueError("client_timeout must be positive and finite")
        if self.size > DEFAULT_TRANSFER_LIMIT and not self.allow_large_sample:
            raise ValueError(f"Samples above {DEFAULT_TRANSFER_LIMIT} require --allow-large-sample")


def build_command(config: SamplingConfig) -> list[str]:
    config.validate()
    pipeline = [{"$sample": {"size": config.size}}, {"$project": dict.fromkeys(FIELDS, 1)}]
    # Explicit prefixes prevent ObjectId/string collisions. Fail closed for
    # other BSON types rather than depend on shell-specific tojson formatting.
    script = (
        f"var source = db.getSiblingDB({json.dumps(config.database)});"
        f"source.getCollection({json.dumps(config.collection)}).aggregate("
        f"{json.dumps(pipeline)}, {{maxTimeMS: {config.server_timeout_ms}}})"
        ".forEach(function(doc) {"
        "var id = doc._id; var encoded;"
        "if (typeof id === 'string') encoded = 'string:' + JSON.stringify(id);"
        "else if (id instanceof ObjectId) "
        "encoded = 'objectid:' + id.valueOf().toLowerCase();"
        "else throw new Error('Unsupported BSON source identifier type');"
        "var row = {source_id: encoded};"
        "['code','smiles','inchikey','formula'].forEach(function(k) {"
        "if (Object.prototype.hasOwnProperty.call(doc,k)) row[k] = doc[k];});"
        "print(JSON.stringify(row));});"
    )
    return ["ssh", "-T", "-o", "BatchMode=yes", "--", config.ssh_target,
            shlex.join(["mongo", "--quiet", "--norc", "--eval", script])]


def _transfer(command: list[str], output, config: SamplingConfig) -> None:
    # Disk-backed stderr avoids deadlocks and unbounded RAM. Only a bounded
    # prefix survives failure; never interpolate server output into exceptions.
    with tempfile.TemporaryFile() as errors:
        try:
            with subprocess.Popen(
                command, stdout=output, stderr=errors, shell=False
            ) as process:
                deadline = time.monotonic() + config.client_timeout
                try:
                    while True:
                        status = process.poll()
                        transferred = (
                            os.fstat(output.fileno()).st_size
                            + os.fstat(errors.fileno()).st_size
                        )
                        if transferred > config.max_transfer_bytes:
                            raise RuntimeError("Transfer byte limit exceeded")
                        if status is not None:
                            if status:
                                raise RuntimeError(
                                    f"SSH/mongo exited with status {status}; "
                                    "extraction not finalized"
                                )
                            break
                        if time.monotonic() >= deadline:
                            raise subprocess.TimeoutExpired(
                                command, config.client_timeout
                            )
                        time.sleep(0.05)
                except BaseException:
                    process.kill()
                    process.wait()
                    raise
        except BaseException as error:
            errors.seek(0)
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=Path(output.name).parent,
                prefix=f".{config.package}.", suffix=".stderr.log", delete=False
            ) as diagnostic:
                # NamedTemporaryFile creates mode 0600 regardless of umask.
                diagnostic.write(errors.read(DIAGNOSTIC_LIMIT))
                diagnostic.flush()
                os.fsync(diagnostic.fileno())
            error.add_note(
                f"Private stderr diagnostic (first {DIAGNOSTIC_LIMIT} bytes): "
                f"{diagnostic.name}. May contain sensitive server details; "
                "inspect locally and redact before sharing."
            )
            raise


def sample_mongo(config: SamplingConfig, *, dry_run: bool = False) -> dict:
    """Never retry. Publish via no-clobber hard links; manifest is completion marker.

    Failures retain uniquely named staging files for explicit inspection/removal.
    If manifest publication fails after data publication, the data has already
    passed integrity; its staging manifest permits manual recovery, without SSH.
    """
    command = build_command(config)
    directory = Path(config.raw_root) / config.database
    destination = directory / f"{config.package}_pool.jsonl"
    metadata = destination.with_suffix(".meta.json")
    if any(os.path.lexists(p) for p in (destination, metadata)):
        raise FileExistsError(f"Destination already exists: {destination} or {metadata}")
    if dry_run:
        return {"command": command, "output": str(destination), "metadata": str(metadata),
                "sampling_method": METHOD, "sampling_method_limitations": LIMITATIONS,
                "requested_record_count": config.size, "max_transfer_bytes": config.max_transfer_bytes}
    directory.mkdir(parents=True, exist_ok=True)
    staged_metadata = None
    started = datetime.now(timezone.utc).isoformat()
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=directory, prefix=f".{config.package}.",
        suffix=".partial", delete=False
    ) as stream:
        staged_data = Path(stream.name)
        try:
            _transfer(command, stream, config)
            stream.flush()
            os.fsync(stream.fileno())
            report = validate_jsonl(staged_data, config.size)
            if not report.success:
                raise ValueError("Extraction integrity failed: " + "; ".join(report.errors[:10]))
            manifest = {
                "manifest_schema_version": 1, "source_database": config.database,
                "source_collection": config.collection, "package_name": config.package,
                "sampling_method": METHOD, "sampling_method_limitations": LIMITATIONS,
                "requested_record_count": config.size,
                "actual_extracted_record_count": report.total_records,
                "extraction_timestamp_utc": started,
                "extraction_completed_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": report.sha256, "output_filename": destination.name,
                "fields_retrieved": FIELDS,
                "source_identity_encoding": SOURCE_ID_ENCODING,
                "extraction_integrity_status": "passed", "missing_smiles_count": report.missing_smiles,
                "server_timeout_ms": config.server_timeout_ms,
                "client_timeout_seconds": config.client_timeout,
                "max_transfer_bytes": config.max_transfer_bytes,
                "source_collection_size": None, "snapshot_identifier": None,
            }
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=directory,
                prefix=f".{config.package}.", suffix=".meta.partial", delete=False
            ) as meta:
                staged_metadata = Path(meta.name)
                json.dump(manifest, meta, indent=2, ensure_ascii=False, allow_nan=False)
                meta.write("\n")
                meta.flush()
                os.fsync(meta.fileno())
            os.link(staged_data, destination)  # atomic and refuses existing files
            os.link(staged_metadata, metadata)  # published last: completion marker
        except BaseException as error:
            error.add_note(
                f"Recovery files: {staged_data}; metadata: {staged_metadata}. "
                "No automatic retry."
            )
            raise
    staged_data.unlink()
    staged_metadata.unlink()
    return manifest
