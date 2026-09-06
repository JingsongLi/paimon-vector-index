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

"""Measure the public Python IVF-SQ reader, including positional-I/O callbacks.

Build the native library in release mode, set PAIMON_VINDEX_LIB_PATH, and put
this checkout's python directory on PYTHONPATH. Uses the same fvecs/ivecs as
ann_bench and benchmark_lance_ivfsq.py. Emit one JSON row per repetition.
"""

import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--nq", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--memory-budget-bytes", type=int, default=4 * 1024**3)
    args = parser.parse_args()
    if min(args.nprobe, args.nq, args.k, args.threads, args.repeats) <= 0 or args.memory_budget_bytes < 0:
        parser.error("counts must be positive; memory budget must be nonnegative")
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    import numpy as np
    from paimon_vindex import SearchParams, VectorIndexReader

    def read_vectors(path, dtype):
        raw = np.memmap(path, mode="r", dtype="<i4")
        width = int(raw[0])
        if width <= 0 or raw.size % (width + 1):
            raise ValueError(f"Invalid vector file: {path}")
        rows = raw.reshape(-1, width + 1)
        if not np.all(rows[:, 0] == width):
            raise ValueError(f"Nonuniform vector dimensions: {path}")
        return np.array(rows[:args.nq, 1:].view(dtype), copy=True)

    queries = read_vectors(args.queries, "<f4")
    truth = read_vectors(args.ground_truth, "<i4")[:, :args.k]
    if len(queries) != args.nq or truth.shape != (args.nq, args.k):
        parser.error("not enough queries or ground-truth neighbors")
    params = SearchParams.ivf(args.k, args.nprobe)

    class FileInput:
        def __init__(self, fd):
            self.fd = fd

        def pread_many(self, ranges):
            return [os.pread(self.fd, length, offset) for offset, length in ranges]

    def recall(ids):
        return sum(len(set(row) & set(expected)) for row, expected in zip(ids, truth)) / truth.size

    for repeat in range(args.repeats):
        with args.index.open("rb") as file:
            with VectorIndexReader(FileInput(file.fileno()), args.memory_budget_bytes) as reader:
                reader.search_batch(queries, params)
                latencies, results = [], []
                for query in queries:
                    started = time.perf_counter()
                    ids, _ = reader.search(query, params)
                    latencies.append(time.perf_counter() - started)
                    results.append(ids)
                batch_s = []
                for _ in range(3):
                    started = time.perf_counter()
                    batch_ids, _ = reader.search_batch(queries, params)
                    batch_s.append(time.perf_counter() - started)
                print(json.dumps({
                    "engine": "paimon-python", "repeat": repeat,
                    "nq": args.nq, "nprobe": args.nprobe, "k": args.k,
                    "threads": args.threads, "memory_budget_bytes": args.memory_budget_bytes,
                    "recall": recall(results), "batch_recall": recall(batch_ids),
                    "p50_ms": float(np.percentile(latencies, 50) * 1000),
                    "p95_ms": float(np.percentile(latencies, 95) * 1000),
                    "sequential_qps": args.nq / sum(latencies),
                    "batch_s": batch_s, "batch_qps": args.nq / float(np.median(batch_s)),
                }), flush=True)


if __name__ == "__main__":
    main()
