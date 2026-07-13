#!/usr/bin/env python3
import os
import random
import sys
from pathlib import Path


def read_matrix_market_graph(path):
    symmetric = False
    pattern = False
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        if header.startswith("%%MatrixMarket"):
            lower = header.lower()
            symmetric = "symmetric" in lower
            pattern = "pattern" in lower
        else:
            f.seek(0)

        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        if not line:
            raise ValueError(f"missing MatrixMarket dimensions in {path}")
        m, n, nnz = [int(x) for x in line.split()[:3]]
        if m != n:
            raise ValueError(f"expected square matrix, got {m} x {n}: {path}")

        adjacency = [set() for _ in range(n)]
        degrees = [0] * n
        for line in f:
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            r = int(parts[0]) - 1
            c = int(parts[1]) - 1
            if r < 0 or r >= n or c < 0 or c >= n:
                raise ValueError(f"entry out of range in {path}: {r + 1}, {c + 1}")
            degrees[r] += 1
            adjacency[r].add(c)
            adjacency[c].add(r)
            if symmetric and r != c:
                degrees[c] += 1
        return n, adjacency, degrees


def write_order(order, path):
    n = len(order)
    seen = [False] * n
    for new_id, old_id in enumerate(order):
        if old_id < 0 or old_id >= n:
            raise ValueError(f"invalid old id {old_id} at new id {new_id}")
        if seen[old_id]:
            raise ValueError(f"duplicate old id {old_id}")
        seen[old_id] = True
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for old_id in order:
            f.write(f"{old_id}\n")


def native_original(input_mtx, output_order):
    n, _, _ = read_matrix_market_graph(input_mtx)
    write_order(list(range(n)), output_order)


def native_random(input_mtx, output_order):
    n, _, _ = read_matrix_market_graph(input_mtx)
    seed = int(os.environ.get("TASA_RANDOM_SEED", "1"))
    order = list(range(n))
    random.Random(seed).shuffle(order)
    write_order(order, output_order)


def native_degree(input_mtx, output_order):
    n, _, degrees = read_matrix_market_graph(input_mtx)
    order = sorted(range(n), key=lambda i: (-degrees[i], i))
    write_order(order, output_order)


def native_rcm(input_mtx, output_order):
    try:
        import numpy as np
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import reverse_cuthill_mckee
    except Exception as exc:
        print(f"missing_dependency: scipy reverse_cuthill_mckee unavailable: {exc}", file=sys.stderr)
        return 42

    n, adjacency, _ = read_matrix_market_graph(input_mtx)
    rows = []
    cols = []
    for i, neigh in enumerate(adjacency):
        for j in neigh:
            rows.append(i)
            cols.append(j)
    data = np.ones(len(rows), dtype=np.int8)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    order = reverse_cuthill_mckee(graph, symmetric_mode=True)
    write_order([int(x) for x in order], output_order)
    return 0


def run_native(kind, argv):
    if len(argv) != 3:
        print(f"usage: {argv[0]} <input.mtx> <output.order>", file=sys.stderr)
        return 2
    input_mtx, output_order = argv[1], argv[2]
    try:
        if kind == "original":
            native_original(input_mtx, output_order)
        elif kind == "random":
            native_random(input_mtx, output_order)
        elif kind == "degree":
            native_degree(input_mtx, output_order)
        elif kind == "rcm":
            return native_rcm(input_mtx, output_order)
        else:
            print(f"unknown native reorder kind: {kind}", file=sys.stderr)
            return 2
    except Exception as exc:
        print(f"order_gen failed for {kind}: {exc}", file=sys.stderr)
        return 1
    return 0
