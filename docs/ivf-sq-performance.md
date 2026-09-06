<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
-->

# IVF-SQ versus Lance: September 2026

SIFT1M and GloVe improve in build time, Python query latency, batch throughput,
and recall relative to Lance 11.0.0. The GIST1M rerun improves build time and
recall, with a modest 5% lower median Python P95, but **batch throughput remains
26% below Lance**. The goal of beating Lance in every measured workload is not
yet met. These are local measurements with the configuration below.

The site pages summarize [current results](ivf-sq.html#benchmarks),
[reader options](api.html#reader-options), and [benchmark setup](development.html#ivfsq-lance).

## Results

Medians of three runs, Apple M4 Pro (12 CPU cores, 48 GiB RAM), eight workers,
release Rust 1.95.0, 1,024 partitions, 64 probes, Top-10, 65,536 training rows,
and the first 1,000 independent ANN-Benchmarks queries. GloVe base/query
vectors were normalized identically before either engine read them.

| Corpus | Engine | Index build | Python query P95 | Python batch QPS | Recall@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| SIFT1M, 1M × 128 | Paimon IVF-SQ | 0.886 s | 0.260 ms | 9,927 | 0.9812 |
| SIFT1M | Lance IVF-SQ 11.0.0 | 3.613 s | 1.099 ms | 4,181 | 0.9772–0.9775 |
| GIST1M, 1M × 960 | Paimon IVF-SQ | 5.850 s | 1.769 ms | 986 | 0.9399 |
| GIST1M | Lance IVF-SQ 11.0.0 | 18.651 s | 1.862 ms | 1,334 | 0.9249 |
| GloVe, 1,183,514 × 100 | Paimon IVF-SQ | 0.797 s | 0.240 ms | 11,052 | 0.8760 |
| GloVe | Lance IVF-SQ 11.0.0 | 2.790 s | 1.089 ms | 4,240 | 0.7843–0.7845 |

In SIFT / GIST / GloVe order, construction is **4.08× / 3.19× / 3.50× as fast**
as Lance, Python P95 is **76% / 5% / 78% lower**, and Python batch throughput
is **2.37× / 0.74× / 2.61×** Lance's. GIST therefore retains a batch-throughput
gap. Its three P95 observations span 1.692–1.857 ms for Paimon and
1.823–2.045 ms for Lance with parallelism 8; the ranges overlap, so the small
median latency advantage should not be treated as a large or universal win.
Lance recall ranges show the medians of the two scheduling modes. Its faster
setting is used separately for each performance metric: partition parallelism
8 for single-query latency and the default 0 for batch throughput. No raw-vector
refinement is requested in either engine. Returned data consists of IDs and
distances; neither benchmark requests the raw vector column.

The [recorded measurements](benchmarks/ivfsq-lance-20260906.json) include all
three repetitions and both Lance scheduling modes. Both engines use their native training algorithms and iteration policies;
training sample counts match, but learned centroids need not. Lance training is not
bitwise deterministic across builds, so recall varies slightly between runs.
Paimon's Rust and Python result paths can choose different members of a tied
boundary; SIFT and GIST recall differ by 0.0001 between native and Python
single-query entry points. GIST Python batch recall is 0.9400. The GIST rerun
uses three interleaved baseline/current/Python/Lance repetitions on the same day.

## Changes and baseline

The baseline is commit `8dcabf208c99707eab3938af0bcc40530fdb4cad`.

| Rust ann_bench metric | SIFT baseline → current | GIST baseline → current | GloVe baseline → current |
| --- | ---: | ---: | ---: |
| Build | 934 → 886 ms | 6,404 → 5,850 ms | 849 → 797 ms |
| Encode/add | 549 → 538 ms | 3,705 → 3,613 ms | 539 → 517 ms |
| Serialize | 125 → 93 ms | 935 → 563 ms | 118 → 90 ms |
| Query P95 | 840 → 298 µs | 4,386 → 1,828 µs | 739 → 282 µs |
| Batch QPS | 6,417 → 8,987 | 899 → 998 | 6,988 → 10,009 |
| Recall@10 | 0.8626 → 0.9811 | 0.8576 → 0.9400 | 0.8036 → 0.8760 |
| File bytes | 131,359,859 → unchanged | 973,626,471 → unchanged | 121,822,316 → unchanged |

GIST serialization time falls by 40%, native query P95 is 58% lower, and native
batch throughput is 11% higher than the same eight-worker baseline. Peak process
RSS falls from 5.08 to 4.87 GiB. These improvements do not close the batch gap to
Lance shown above.

1. **Avoid sparse-partition clipping.** Training previously estimated a separate
   minimum and maximum from each partition's often tiny sample. Constant sample
   dimensions and narrow extrema clipped unseen residuals. The trainer now pools
   per-dimension residual extrema across partitions. Holding centroids fixed,
   this alone raised SIFT Recall@10 from about 0.863 to 0.981. Broad pooled bounds
   can reduce resolution on corpora with extreme outliers; measure such data
   before adopting this training policy.
2. **Fuse encoding and bound training.** Partition-local reductions avoid the
   training residual matrix. Encoding computes inverse scales once per partition,
   subtracts the centroid in registers, and packs rounded NEON/AVX2 results
   directly into the destination. Serialization transposes partitions in parallel
   within 16 MiB batches (one oversized partition is processed alone).
3. **Reuse query heaps and prune safely.** Batch scans retain one heap per query.
   An L2 block can stop after its first half only if every nonnegative partial
   distance already exceeds or equals the current cutoff. Competitive distances
   are still evaluated completely. A first partition supplies the cutoff for
   parallel single-query scans. A one-query batch uses the single-query path.
4. **Reuse decoded partitions within the existing memory budget.** The unified
   reader and bindings use a bounded FIFO cache. Cache hits share immutable
   buffers with `Arc`, bypass positional I/O, and avoid decoding IDs again.
   Metadata, cache slot/queue storage, and retained payload capacities are
   reserved before retaining entries. Filters and distances are never cached.
   Oversized streamed partitions bypass the cache. Zero budget disables caching;
   required metadata still loads. Direct `IVFSQIndexReader::open` stays uncached;
   `open_with_options` enables the cache.

The SIFT/GloVe uncached optimized intermediate also improved Rust batch
throughput to 9,176 / 10,224 QPS at the new higher recall; this intermediate
was not rerun on GIST. Cache-enabled ann_bench batch
numbers are slightly lower because it opens a fresh reader and times a complete
first batch, including payload reads and cache insertion. The Python comparison
warms the selected partitions before timing both engines. Its native file adapter
uses ordinary `os.pread` callbacks; no special in-memory adapter is used.

Paimon's configured reader budget is 4 GiB and Lance's index cache is 1 GiB
(explicit `index_cache_size_bytes`, not the deprecated entry-count parameter);
each entire measured index fits within either budget. Neither reserves that
whole amount as payload memory. Paimon's peak process RSS was about
787 / 4,986 / 733 MiB for SIFT / GIST / GloVe, including the benchmark's source data and build phases. Build time includes
training, assignment, encoding and index serialization. Lance additionally needs
a source dataset; its data-writing time is recorded separately and **excluded**
from the comparison above. Paimon build stages use the Rust benchmark; query
latency/throughput in the first table use both public Python interfaces.

The IVSQ v1 format, flags, row-ID encoding, and golden fixture bytes remain
compatible. Old files benefit from the reader changes immediately. Rebuilding
is required to get the new training bounds; existing files keep their recorded
per-partition quantizers.

## Reproduce

Obtain the public SIFT, GIST, and GloVe HDF5 files from
[ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks). Install `numpy`,
`h5py`, `pyarrow`, and `pylance==11.0.0` in a temporary environment. Run each
benchmark separately, without concurrent compilation or other benchmarks. Run from
the repository root, and set `DATA` to the directory containing the downloaded
HDF5 files and `OUT` to the directory for generated indexes before running the
commands below.

```sh
python tools/convert_ann_benchmarks.py "$DATA/sift-128-euclidean.hdf5" "$DATA/sift" \
  --prefix sift --query-limit 1000
python tools/convert_ann_benchmarks.py "$DATA/gist-960-euclidean.hdf5" "$DATA/gist" \
  --prefix gist --query-limit 1000
python tools/convert_ann_benchmarks.py "$DATA/glove-100-angular.hdf5" "$DATA/glove" \
  --prefix glove --query-limit 1000 --normalize-l2
cargo bench -p paimon-vindex-core --bench ann_bench --no-run
cargo build --release -p paimon-vindex-ffi

# Repeat for corpus=gist and corpus=glove, and repeat each engine three times.
corpus=sift
export RAYON_NUM_THREADS=8 ANN_INDEXES=IVF_SQ ANN_TRAIN_N=65536
export ANN_NLIST=1024 ANN_NPROBE=64 ANN_K=10 ANN_STORAGE_CASES=local_ssd_warm_cache
export ANN_BASE_FVECS="$DATA/$corpus/${corpus}_base.fvecs"
export ANN_QUERY_FVECS="$DATA/$corpus/${corpus}_query.fvecs"
export ANN_GROUND_TRUTH_IVECS="$DATA/$corpus/${corpus}_ground_truth.ivecs"
export ANN_KEEP_INDEXES=1 ANN_OUTPUT_DIR="$OUT/paimon-$corpus"
cargo bench -p paimon-vindex-core --bench ann_bench

python tools/benchmark_lance_ivfsq.py \
  --base "$ANN_BASE_FVECS" --queries "$ANN_QUERY_FVECS" \
  --ground-truth "$ANN_GROUND_TRUTH_IVECS" --output-dir "$OUT/lance-$corpus" \
  --threads 8 --train-n 65536 --nlist 1024 --nprobe 64 --nq 1000 --k 10 \
  --query-parallelism 0 8 --cache-bytes 1073741824 --repeats 3

# INDEX is the ivf_sq.index preserved under ANN_OUTPUT_DIR/<pid>/.
PYTHONPATH=python PAIMON_VINDEX_LIB_PATH="$PWD/target/release/libpaimon_vindex_ffi.dylib" \
python tools/benchmark_ivfsq_reader.py --index "$INDEX" \
  --queries "$ANN_QUERY_FVECS" --ground-truth "$ANN_GROUND_TRUTH_IVECS" \
  --threads 8 --nprobe 64 --nq 1000 --k 10 \
  --memory-budget-bytes 4294967296 --repeats 3
```

Use `--memory-budget-bytes 0` on the Python reader benchmark to isolate the
uncached scan path; this does not flush the operating-system cache. For IVF-SQ,
`optimize_for_search` and `warmup_queries` initialize metadata only. Replay actual
searches to warm the partition cache, as the Python script does. On Linux, use the `.so` native library instead of `.dylib`.
The Lance script creates a new dataset per repetition and reports the path;
remove those generated benchmark outputs when no longer needed.

## Verification and limits

- Workspace tests: 510 passed, 2 intentionally ignored; includes v1 golden fixtures.
- Python bindings: 28 passed.
- The pre-change reader at `8dcabf2` opened newly generated SIFT, GIST, and
  GloVe indexes and completed 1,000 single queries plus batch search per corpus.
  Recall@10 differed from the new reader by at most 0.0002; this verifies file
  compatibility, not identical ordering of every result.
- x86_64 build and SQ tests under Rosetta: 33 passed. Rosetta reported AVX2/FMA
  unavailable, so AVX2 was compiled but its runtime kernel needs native x86 CI.
- `cargo fmt`, workspace Clippy with warnings denied, license headers, and diff
  whitespace checks passed.
- Added regression coverage for packed-encoding rounding/tails, residual extrema,
  sparse/empty partition calibration, L2 cutoff boundaries, single-query and
  seeded batch paths, cache reuse, eviction, budget bypass, filtering, and read
  failures followed by retries.

This run does not establish superiority on Linux/x86, cold storage,
object-store latency, or every metric/distribution. GIST batch throughput remains
below Lance. The homepage build/local tables contain the refreshed IVF-SQ rows;
other indexes retain their labeled July measurements, and the historical remote
models and implementation notes remain in collapsible archives.
