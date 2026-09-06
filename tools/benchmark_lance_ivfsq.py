#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Benchmark Lance IVF-SQ against the exact fvecs/ivecs used by ann_bench.

Requires pylance 11+, numpy, and pyarrow. Output is JSONL, one row per
build/query configuration. Run independently of other CPU benchmarks.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--train-n", type=int, default=65536)
    parser.add_argument("--nprobe", type=int, nargs="+", default=[64])
    parser.add_argument("--query-parallelism", type=int, nargs="+", default=[0, 8])
    parser.add_argument("--nq", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cache-bytes", type=int, default=1024**3)
    args = parser.parse_args()
    if min(args.nlist, args.train_n, args.nq, args.k, args.threads, args.repeats) <= 0:
        parser.error("counts must be positive")
    if args.train_n % args.nlist or args.train_n < args.nlist:
        parser.error("train-n must be a positive multiple of nlist")
    if any(p <= 0 or p > args.nlist for p in args.nprobe):
        parser.error("nprobe must be in 1..nlist")
    if args.cache_bytes < 0 or any(p < 0 for p in args.query_parallelism):
        parser.error("cache size and query parallelism must be nonnegative")
    # Set pools before importing either native runtime.
    for key in ["LANCE_CPU_THREADS", "LANCE_IO_THREADS", "RAYON_NUM_THREADS"]:
        os.environ[key] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import lance
    import numpy as np
    import pyarrow as pa

    def read_vectors(path, dtype):
        raw = np.memmap(path, mode="r", dtype="<i4")
        width = int(raw[0])
        if width <= 0 or raw.size % (width + 1):
            raise ValueError(f"Invalid vector file: {path}")
        rows = raw.reshape(-1, width + 1)
        if not np.all(rows[:, 0] == width):
            raise ValueError(f"Nonuniform vector dimensions: {path}")
        return np.array(rows[:, 1:].view(dtype), copy=True)

    base = read_vectors(args.base, "<f4")
    queries = read_vectors(args.queries, "<f4")[:args.nq]
    truth = read_vectors(args.ground_truth, "<i4")[:args.nq, :args.k]
    if len(queries) != args.nq or truth.shape != (args.nq, args.k):
        parser.error("not enough query/ground-truth rows or ground-truth neighbors")
    if base.shape[1] != queries.shape[1] or args.train_n > len(base):
        parser.error("inconsistent dimensions or train-n exceeds the base")
    if not np.isfinite(base).all() or not np.isfinite(queries).all():
        parser.error("vectors must be finite")
    if np.any(truth < 0) or np.any(truth >= len(base)):
        parser.error("ground-truth IDs must refer to base rows")
    table = pa.table({"vector": pa.FixedSizeListArray.from_arrays(
        pa.array(base.ravel()), base.shape[1])})
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def recall(results):
        return sum(len(set(r) & set(g)) for r, g in zip(results, truth)) / truth.size

    for repeat in range(args.repeats):
        path = Path(tempfile.mkdtemp(prefix="lance-ivfsq-", dir=args.output_dir)) / "data.lance"
        started = time.perf_counter()
        dataset = lance.write_dataset(table, str(path))
        data_write_s = time.perf_counter() - started
        started = time.perf_counter()
        dataset.create_index("vector", "IVF_SQ", metric="L2",
                             num_partitions=args.nlist,
                             sample_rate=args.train_n // args.nlist)
        index_build_s = time.perf_counter() - started
        dataset = lance.dataset(str(path), index_cache_size_bytes=args.cache_bytes)
        index_bytes = sum(p.stat().st_size for p in (path / "_indices").rglob("*") if p.is_file())
        for nprobe in args.nprobe:
            for parallelism in args.query_parallelism:
                def search(q):
                    return dataset.to_table(columns=["_distance"], with_row_id=True,
                                            nearest={"column": "vector", "q": q,
                                                     "k": args.k, "nprobes": nprobe,
                                                     "query_parallelism": parallelism})
                # Warm the union of selected partitions. Leave refine_factor unset:
                # returning IDs/distances must not fetch or rerank raw vectors.
                search(queries)
                latencies, results = [], []
                for query in queries:
                    started = time.perf_counter()
                    result = search(query)
                    latencies.append(time.perf_counter() - started)
                    results.append(result["_rowid"].to_numpy())
                batch_s = []
                for _ in range(3):
                    started = time.perf_counter()
                    batch = search(queries)
                    batch_s.append(time.perf_counter() - started)
                query_index = batch["query_index"].to_numpy()
                row_ids = batch["_rowid"].to_numpy()
                batch_results = [row_ids[query_index == i] for i in range(args.nq)]
                print(json.dumps({
                    "engine": "lance", "version": lance.__version__,
                    "machine": platform.machine(), "repeat": repeat,
                    "base": str(args.base), "n": len(base), "d": base.shape[1],
                    "nq": args.nq, "k": args.k, "nlist": args.nlist,
                    "train_n": args.train_n, "threads": args.threads,
                    "nprobe": nprobe, "query_parallelism": parallelism,
                    "data_write_s": data_write_s, "index_build_s": index_build_s,
                    "index_bytes": index_bytes, "recall": recall(results),
                    "p50_ms": float(np.percentile(latencies, 50) * 1000),
                    "p95_ms": float(np.percentile(latencies, 95) * 1000),
                    "sequential_qps": args.nq / sum(latencies),
                    "batch_s": batch_s, "batch_qps": args.nq / float(np.median(batch_s)),
                    "batch_recall": recall(batch_results), "index_path": str(path),
                }), flush=True)


if __name__ == "__main__":
    main()
