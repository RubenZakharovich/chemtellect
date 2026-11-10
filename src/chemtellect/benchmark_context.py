"""Explicit, synchronous context sampling outside short retrieval timers."""
from time import perf_counter_ns

import psutil

from chemtellect.fingerprint_store import utc


def memory_snapshot(stage):
    start = perf_counter_ns()
    sampled_utc = utc()
    ram = psutil.virtual_memory()
    rss = psutil.Process().memory_info().rss
    return {'stage': stage, 'sampled_utc': sampled_utc, 'sampling_start_perf_counter_ns': start,
            'sampling_duration_ns': perf_counter_ns() - start,
            'system_ram_total_bytes': int(ram.total), 'system_ram_available_bytes': int(ram.available),
            'process_rss_bytes': int(rss), 'interpretation': 'synchronous snapshots; neither is true peak memory'}
