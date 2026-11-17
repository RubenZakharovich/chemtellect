# Efficiency Evaluation of HNSW-based Approximate Molecular Retrieval

Objective: Evaluating recall and computational efficiency of HNSW-based approximate molecular retrieval against exhaustive fingerprint similarity search, using different similarity thresholds and candidate counts.

Given a dataset of $N$ molecules and a fingerprint representation for each molecule. For every query molecule $q$, we conduct two searches:

1. Compare the query fingerprint against every molecule in the dataset (exhaustive search).
2. Use HNSW approximate search to retrieve a smaller candidate set.

This is performed against a small representative dataset (a random sample) as it makes exhaustive search computationally affordable.

Finalize by comparing the retrieved sets. 

*For a chosen similarity threshold, how many of the molecules found by exhaustive search were recovered through HNSW?*

>Hypothesis to test: Higher similarity thresholds may lead to better threshold recall because fewer molecules satisfy the threshold condition. the composition of the retrieved set also matters. In particular, changing an evaluation threshold does not change a previously obtained HNSW candidate set.

*For a chosen number of $k$ nearest neighbors, how many of the $k$ most similar molecules identified by exhaustive search were also retrieved by HNSW?*

Both questions above measure set overlap rather than the ordering of retrieved molecules.

**Evaluation Criteria**:
- *Explore how HNSW parameter settings affect retrieval.*
- *Explore how retrieval quality varies with different similarity thresholds and candidate counts.*
- *Measure performance: retrieval time, query latency, index construction time, memory consumption, and how these change as the dataset grows.*

# Definitions

The following two sets are constructed for the experiment:

The ground-truth set for threshold $\tau$ in indexed reference records $D$ is
$G_\tau(q)=\{x\in D:T(q,x)\geq\tau\}$

where $T(q,x)$ is the similarity function.

The HNSW candidate set is
$C_k(q)=\operatorname{HNSW}(q,k)$

where $k$ is the number of candidates requested.

### Threshold recall evaluation metric

${ R_\tau(q)= \frac{|C_k(q)\cap G_\tau(q)|} {|G_\tau(q)|} }$
For a nonempty ground-truth set and a budget of $k$ candidates the maximum attainable threshold recall is 
${R_\tau^{\max}(q)=\min\left(1,\frac{k}{|G_\tau(q)|}\right)}$

This is recall at similarity threshold $\tau$, or the proportion of truly qualifying molecules retrieved by HNSW.

### Top-$k$ recall evaluation metric
Let $E_k(q)$ be the set of the $k$ most similar molecules according to exhaustive Tanimoto search, and $C_k(q)$ the $k$ candidates returned by HNSW. Then top-$k$ recall is

${R@k(q)=\frac{|E_k(q)\cap C_k(q)|}{k}}$

This measures how much the two sets overlap, not their internal ordering. When several molecules tie at the exact $k^{th}$ similarity score, we'll use a tie-aware version to avoid penalizing equally good alternatives.

Example, if exact top-5 is `{A,B,C,D,E}` and HNSW returns `{A,B,C,F,G}`, `R@5 = 3/5 = 60%`.

Neither top-k recall nor threshold recall measures the ordering of the returned records.

### Tie-aware top-k recall
Tanimoto similarities often tie, especially with fixed-length fingerprints. If several records tie at the exact $k^{th}$ score, an equally similar record chosen instead of the arbitrarily selected $k^{th}$ record should not be counted as an error.

Let $s_k$ be the $k^{th}$-largest exact score, $H = \{x \in D : T(q,x) > s_k\}$ and 
$B = \{x \in D : T(q,x) = s_k\}$. Then the number of acceptable hits is:
$h_{\mathrm{tie}}=|C_k\cap H|+\min(k-|H|,\ |C_k\cap B|)$
and 
$\operatorname{TieAwareRecall}@k(q) = \frac{h_{\mathrm{tie}}(q)}{k}$

This rewards all exact neighbors strictly above the cutoff and allows 
any $k - |H|$ equally scoring records at the boundary to fill the remaining positions. Use this as the primary top-k metric, optionally report raw ID-overlap recall as supplemental. Use exactly the computed similarity scores rather than an arbitrary rounding rule for tie determination, or document a consistent tolerance if floating-point representations require one.

### How to interpret the two recalls
| Top-k recall | Threshold recall | Interpretation                                                                                                     |
| ------------ | ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| High         | High             | HNSW identifies nearest neighbors and covers most threshold-relevant molecules.                                    |
| High         | Low              | HNSW finds most top-k neighbors, but `                                                                             |
| Low          | High             | HNSW includes all few high-threshold relevant molecules, but misses some lower-scoring members of the exact top-k. |
| Low          | Low              | HNSW misses many nearest and threshold-relevant molecules; compare against the attainable ceiling and tie policy.  |

# Distance Functions

As mentioned above, $T(A,B)$ is the similarity function.

For this evaluation the binary Morgan fingerprints and Tanimoto similarity is chosen as initial, test case without claiming that these are necessarily the optimal molecular representations.

For binary fingerprints, Tanimoto similarity is

$T(A,B)= \frac{|A\cap B|}{|A\cup B|}$

where $A$ and $B$ represent the sets of active fingerprint bits.

The corresponding Jaccard distance is
$\boxed{d_J(A,B)=1-T(A,B)}$

Minimizing Jaccard distance is exactly equivalent to maximizing binary Tanimoto similarity.

# HNSW parameters
The parameters below determine how HNSW builds and navigates its search graph.

| Parameter        | Purpose                                                 | Trade-off                                                       |
| ---------------- | ------------------------------------------------------- | --------------------------------------------------------------- |
| `M`              | Controls graph connectivity                             | More connections increase memory use but can improve recall     |
| `efConstruction` | Controls candidate exploration while building the graph | More thorough construction is slower                            |
| `efSearch`       | Controls candidate exploration during queries           | Higher values generally improve recall but increase search work |

# Starting configuration of the variables.

These are starting configurations, not optimized settings or final conclusions.

| Variable                    | Initial experimental values              |
| --------------------------- | ---------------------------------------- |
| Dataset size                | 1,000 molecules, then 10,000 and 100,000 |
| Fingerprint                 | Morgan, radius 2, 2,048 bits             |
| Ground-truth metric         | Binary Tanimoto                          |
| Similarity threshold $\tau$ | 0.3, 0.5, 0.7, 0.9                       |
| Retrieved candidates $k$    | 10, 25, 50, 100                          |
| HNSW metric                 | Jaccard distance                         |
| HNSW `M`                    | 8, 16, 32                                |
| HNSW `efSearch`             | 20, 50, 100, 200                         |
| Query molecules             | 100 sampled from the dataset             |

Edge Cases:
- Query molecules should not be part of the indexed dataset, and duplicate structures should be Identified and removed. 
- Account for molecules that satisfy threshold after candidate limit is reached. If 80 molecules satisfy a threshold but search requests only 20 HNSW candidates, the maximum possible recall is 25%, regardless of how accurately the graph navigates.
- Separate retrieval from re-ranking. When the ANN metric differs from Tanimoto, we should measure both candidate generation and exact Tanimoto re-ranking costs.

Details left out:
- Is RDKit's implementation of Morgan fingerprint algorithm any different from other implementations. If yes, what is the most sufficient one to choose.
- Is Morgan fingerprint with radius 2 well suited compared to radius 0 and 1. What are other options other than Morgan representation?
- What are molecule representation options other than binary. Current case considers fingerprints that are fixed-length binary Morgan fingerprints, one other trivial option is count-based fingerprints.

# Benchmarking tools
| Measurement | What it answers                                                  | Proposed tool                      |
| ----------- | ---------------------------------------------------------------- | ---------------------------------- |
| Recall      | What proportion of qualifying molecules did HNSW retrieve?       | NumPy and our own evaluation logic |
| Latency     | How long does each retrieval method take?                        | Python's `time.perf_counter_ns()`  |
| Memory      | How much RAM do our fingerprint collections and indexes consume? | `psutil`                           |

## Experimental Protocol v1.0

The following parameters were used for the first experiment. These are deliberately modest settings.

| Field                        | Pilot value                                                    |
| ---------------------------- | -------------------------------------------------------------- |
| Index size per package       | 1,000 validated records                                        |
| Held-out queries per package | 100                                                            |
| Initial raw package target   | 1,100 usable records                                           |
| Query sampling seed          | 42                                                             |
| Fingerprint                  | New binary Morgan; radius 2; 2,048 bits                        |
| Exact baseline               | Exhaustive binary Tanimoto against all indexed records         |
| Approximate backend          | NMSLIB 2.1.2 HNSW, `bit_jaccard`, `OBJECT_AS_STRING`           |
| HNSW `M`                     | 16                                                             |
| HNSW `efConstruction`        | 200                                                            |
| HNSW `efSearch`              | 200                                                            |
| Candidate counts $k$         | 10, 25, 50, 100                                                |
| Thresholds $τ$               | 0.3, 0.5, 0.7, 0.9                                             |
| Query/index record overlap   | None under holdout split                                       |
| Empty threshold ground truth | Undefined recall; count and report                             |
| Top-k boundary ties          | Tie-aware scoring                                              |
| Threshold evaluation         | Reuse fixed candidates for each $k$/query/search configuration |

There are four additional rules.

1. Empty ground-truth sets: If no other molecule meets a threshold, that query's threshold recall is undefined. We'll exclude it from the mean recall for that threshold and report how frequently this happens.

2. Equal-similarity ties: Different molecules can have identical fingerprint similarity scores. We'll use a tie-aware top-k evaluation to avoid penalizing HNSW simply for returning an equally similar molecule.

3. Fixed candidate pools: For a given $k$, changing $\tau$ will only change the evaluation, not trigger another HNSW search.

4. Timing: Fingerprint generation and index construction will be measured separately from query latency. We'll also record the Python and library versions associated with a benchmark.

# Methodology and Findings

The study asks: **How much faster is approximate HNSW retrieval than exact retrieval, and how many correct neighbors does it miss?**
### Molecule identity, fingerprints, and collisions

A **query** is a molecule whose neighbors we seek. The **index** is the collection of molecules it is allowed to retrieve.

SMILES can describe the same structure using different strings. The preparation workflow parses and sanitizes molecules and produces canonical, isomeric SMILES—a consistent representation under the approved RDKit procedure. Records with the same canonical representation are grouped, retaining the first physical record as representative. No additional chemical standardization is performed.

No canonical structure appears in both query and index groups. A query therefore cannot receive an easy self-match by finding its own canonical structure in the index.

Each structure is represented by an approved Morgan fingerprint: radius 2, 2,048 bits, chirality enabled. These bits summarize structural features; they are not a complete molecular description. All selected fingerprints must be nonzero.

Different canonical structures can have identical fingerprints. The finite fingerprint representation cannot distinguish them. This is a **fingerprint collision**, distinct from a duplicate canonical structure.

**What the code does with collisions:** it preserves the records and their distinct IDs. It does not merge or discard them because their bits match. Each retains its own matrix column and can occupy its own result position. Identical nonzero fingerprints score 1 against each other and have identical scores against every other query. If a query collides with a different indexed structure, a similarity-1 result is permitted despite self-exclusion.

Collisions and other equal-score cases are handled by the [[Efficiency Evaluation of HNSW-based Approximate Molecular Retrieval#Definitions#Tie-aware top-k recall|tie-aware recall]] rule. Returning the same ID twice is invalid; returning two distinct IDs with identical fingerprints is valid.

#### Example
The key distinction is a molecule’s identity versus its fingerprint. Two different molecules can have the same fingerprint because the fingerprint is a compressed description of structural features.

Consider this simplified example. Real fingerprints have 2,048 bits; these have only eight:

|Record ID|Molecular structure|Fingerprint|
|---|---|---|
|Q|Structure A|`10110100`|
|17|Structure B|`10110100`|
|42|Structure C|`10110100`|
|63|Structure D|`10100100`|

A, B, and C are different structures, but the fingerprint cannot distinguish them. This is a fingerprint collision.

##### Why can Q retrieve records 17 and 42 despite self-exclusion?

Self-exclusion removes Q’s own canonical structure, Structure A, from the searchable index.

It does not remove every structure with the same fingerprint. Records 17 and 42 represent different canonical structures, so they remain eligible.

Their Tanimoto similarity to Q is:

$T(Q,17)=T(Q,42)=\frac{4\text{ shared set bits}}{4\text{ bits in the union}}=1$

Thus, similarity 1 means identical fingerprints—not necessarily identical molecules.
##### What does “each retains its own matrix column” mean?

The similarity matrix stores a score for every indexed record, including records whose fingerprints match:

|Query|Record 17|Record 42|Record 63|
|---|--:|--:|--:|
|Q|1.00|1.00|0.75|

The code does not combine columns 17 and 42 into one column. They correspond to two distinct searchable records.

For any other query X, records 17 and 42 will also receive equal scores:
$T(X,17)=T(X,42)$
because the similarity calculation sees exactly the same bits for both records. Those scores need not be 1; they simply must equal each other.

##### How does this affect the returned results?**

For the example above:

|Request|Returned IDs|Interpretation|
|---|---|---|
|Top-1|`[17]`|Fully correct|
|Top-1|`[42]`|Equally correct|
|Top-2|`[17, 42]`|Fully correct: two distinct records|
|Top-2|`[17, 17]`|Invalid: the same record was returned twice|
|Top-2|`[17, 63]`|Only one of the two best records was recovered|

For top-1, the tie-aware rule accepts either 17 or 42. It does not penalize HNSW merely because RDKit’s saved, deterministically ordered answer chose the other ID.

For top-2, both 17 and 42 belong in the answer. Returning 17 and 63 earns:
$R_2=\frac{1}{2}=50\%$
##### Does every equal-score tie indicate a fingerprint collision?**

No. Different fingerprints can also have the same Tanimoto score against a particular query.

A collision means the fingerprints themselves are identical. A score tie means their similarity scores against this query are equal. The tie-aware evaluation handles both.

The practical consequence is that this experiment retrieves molecular records, and multiple result positions may contain records with identical fingerprints. It does not require the returned set to contain k distinct fingerprints or k structurally diverse molecules.

### Constructing the similarity matrix and ground truth

For fingerprints A and B, bit-vector Tanimoto similarity is:

$
T(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$
Before comparative timing, RDKit compares every query with every indexed fingerprint. For query q and indexed record i, it saves:

$
S_{q,i}=T(q,i).
$

Each matrix row contains one query's scores against the entire index; each column refers to one indexed record. A small illustrative matrix might be:

| Query / indexed record |    A |    B |    C |    D |
| ---------------------- | ---: | ---: | ---: | ---: |
| Query 1                | 0.20 | 0.80 | 0.50 | 0.80 |
| Query 2                | 0.70 | 0.10 | 0.40 | 0.30 |

For Query 1, exact top-2 membership is {B, D}. At threshold 0.5, the full qualifying set is {B, C, D}. For Query 2, top-2 membership is {A, C}. These answers come directly from the corresponding row.

| Index size |    Matrix shape | Number of scores |
| ---------- | --------------: | ---------------: |
| 1K         |     100 × 1,000 |          100,000 |
| 10K        |    100 × 10,000 |        1,000,000 |
| 100K       |   100 × 100,000 |       10,000,000 |
| 1M         | 100 × 1,000,000 |      100,000,000 |
Scores are stored as float64 in `exact_scores.npy`, with ordered identities in `records.json` and saved top-k evidence in `ground_truth_top_k.json`. At 100K, the score values alone occupy about 80 MB in decimal units.

Each size receives a fresh reference calculation. Adding molecules adds columns and can change the best k results and threshold membership. Scores between an unchanged query and an unchanged indexed fingerprint remain the same.

Ground truth refers to the results obtained by comparing a query molecule against every molecule in the dataset using our chosen fingerprint representation and Tanimoto similarity measure. It establishes which records should be retrieved, not biological activity, usefulness, or complete structural equivalence.

The saved matrix is used only for validation and evaluation. RDKit does the full search again every time we measure its speed, even though we already saved the correct answers for evaluation. 

### Search methods and correctness checks

| Method | How it serves the experiment                                                                                        |
| ------ | ------------------------------------------------------------------------------------------------------------------- |
| RDKit  | Exhaustive scoring followed by exact top-k selection                                                                |
| FPSim2 | Exact-search implementation using the approved saved fingerprint bits; returned membership is checked against RDKit |
| HNSW   | Approximate graph search; returned membership is measured against RDKit                                             |
HNSW uses `NMSLIB` 2.1.2, `bit_jaccard` distance. Parameters: `M = 16`, `efConstruction = 200`, and `efSearch = 200`. 
`M` controls graph connectivity; `efConstruction` and `efSearch` control construction and search exploration budgets. One construction thread and sequential single-query calls are used. No tuning sweep was performed.

HNSW must return exactly k distinct, valid IDs and finite distances between 0 and 1. Each returned distance is checked against `1 − RDKit similarity`, with absolute tolerance 0.000001 and relative tolerance zero. This tolerance checks native numerical compatibility; it never changes ground-truth membership or recall. Missing good neighbors lowers HNSW recall rather than failing this compatibility check.

### Retrieval quality: per-query recall and mean recall

##### Top-k recall
Recall that the default policy for [[Efficiency Evaluation of HNSW-based Approximate Molecular Retrieval#Top-$k$ recall evaluation metric|Top-k recall]] used is the [[Efficiency Evaluation of HNSW-based Approximate Molecular Retrieval#Definitions#Tie-aware top-k recall|Tie-aware recall]] which asks: How many of the best k neighbors did the method recover?

$
R_k(q)=\frac{|C_k\cap H|+\min(k-|H|,\ |C_k\cap B|)}{k}
$
Ties use direct `float64` equality, with no epsilon. Meaning, we consider scores tied only when their saved `float64` values compare exactly equal. Epsilon would be a small allowance used to treat nearby values as equal.

Saved exact top-k lists break ties by original index position. Meaning, the code orders molecules by:

1. Higher similarity values first.
2. For equal similarity values, smaller original index position first.

Example:  If a list of three records should be retrieved from the set $\{A,B,C,D\}$ where $B,C$ and $D$ have the same similarity value, the code prioritizes $\{A,B,C\}$. $B$ and $C$ win this tie-breaking step because their index positions precede $D$’s, even though all three have the same similarity value. Simply put, $B$ and $C$ come before $D$.

Mean recall is the arithmetic average of the per-query recalls:

$
\overline{R}_k=\frac{1}{Q}\sum_{q=1}^{Q}R_k(q).
$
where $Q$ is the number of queries.

For example, recalls of $100\%$, $90\%$, and $80\%$ average to $90\%$. This does not mean each query achieves $90\%$. The report also shows median and minimum recall to expose some variation between queries. Recall evaluates membership, not the internal ordering of results.

##### Threshold recall

[[Efficiency Evaluation of HNSW-based Approximate Molecular Retrieval#Threshold recall evaluation metric|Threshold recall]] asks: Of all indexed molecules scoring at least $τ$, how many appeared in the returned top-k set? For $τ$ in $\{0.3, 0.5, 0.7, 0.9\}$:

$
G_\tau(q)=\{i:S_{q,i}\ge\tau\},\qquad
R_{\tau,k}(q)=\frac{|C_k(q)\cap G_\tau(q)|}{|G_\tau(q)|}.
$

Comparison is inclusive (`>=`) against the float64 threshold, with no tolerance. Every threshold uses the same designated candidate set; there is no additional retrieval for each threshold.

If $200$ molecules qualify but $k = 10$, even perfect top-$10$ retrieval reaches at most $5\%$ threshold recall. The capacity ceiling is:

$
R_{\tau,k}^{\max}(q)=\min\left(1,\frac{k}{|G_\tau(q)|}\right).
$

### Search repetitions and timing boundaries

Each dataset is tested at $k\in\{10,25,50,100\}$. For every method and $k$, all $100$ queries receive one untimed warm-up pass and five measured passes. Queries execute sequentially. Each $k$ is an independent search, not a truncated top-$100$ result.

For every dataset size, method, and k:

1. Run all $100$ queries once as an untimed warmup.
2. Run all $100$ queries five more times, recording retrieval times.

Thus, for one method at one k:

$100\text{ queries}\times5\text{ measured passes} =500\text{ timed requests}$

The totals are:

|Scope|Timed retrieval requests|
|---|--:|
|One method, one k, one dataset|500|
|One method, all four k values, one dataset|2,000|
|All three methods, one dataset|6,000|
|All three methods, all four dataset comparisons|**24,000**|

There were also **4,800 warmup requests** across the four comparisons, plus separate correctness-check searches.

Each $k$ was searched independently. A top-$100$ result was not reused to manufacture the top-$10$, top-$25$, and top-$50$ timing results.

Within each pass, all methods received the same shuffled query order. Backend execution order also rotated. These controls reduce order-related effects, although they do not eliminate operating-system scheduling or cache effects.

**Recall uses only the first measured pass's candidates.** Later passes provide timing and diagnostics; their results are neither pooled nor selected for best recall.

Retrieval timers include exhaustive scoring plus selection/ordering for RDKit, or native search plus result normalization/ordering for FPSim2 and HNSW. HNSW's internal parsing of its prepared query string is included. 

Fingerprint generation, query preparation, index preparation, graph construction, reference calculation, validation, recall evaluation, writing, and plotting are outside retrieval latency. The new comparison records several separate preparation and evaluation costs, including outside-timer validation/result assembly. Component timings may overlap enclosing stages and must not all be summed. A reported latency describes an already-prepared query and in-memory index, not complete application response time.

### From repeated timings to headline speed
First take the median of five measured times for each single query:

$
m_q=\operatorname{median}(t_{q,1},\ldots,t_{q,5}).
$

Then take the median of those 100 per-query medians:

$
L=\operatorname{median}(m_1,\ldots,m_{100}).
$

This L is the headline latency: the typical query's typical time. For example, five times of 1.30, 1.34, 1.31, 2.10, and 1.33 ms yield a per-query median of 1.33 ms.

The reported mean and p95 are also calculated across the 100 per-query medians, using linear percentile interpolation.

Table below summarizes what do the reported mean and p95 describe.  
They use the same 100 per-query medians:

| Statistic | Calculation                           | Interpretation                          |
| --------- | ------------------------------------- | --------------------------------------- |
| Median$L$ | Middle of the 100 medians             | Typical query’s typical time            |
| Mean      | Sum of the 100 medians divided by 100 | Average representative time             |
| p95       | 95th percentile of the 100 medians    | Upper end of representative query times |
> This p95 describes relatively slow queries after their repeated timings have been summarized.

It does not describe the slowest 5% of all individual requests. A query with four fast repetitions and one very slow repetition can still have a fast median. Consequently, this p95 reduces the influence of occasional timing spikes.

For a baseline B and method H:

$
\text{speedup}=L_B/L_H,\qquad
\text{time saved}=L_B-L_H.
$

A speedup above 1 favors H; below 1 favors B. This is a ratio of aggregate latencies, not a mean of individual query speedups.

Suppose the baseline is FPSim2 and the method being evaluated is HNSW:
