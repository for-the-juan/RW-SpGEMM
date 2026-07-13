#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REORDERING_ROOT = Path(os.environ.get(
    "REORDERING_ROOT",
    "/home/hangcheng.dong/PPoPP27/reordering-spgemm",
))
PARTITION_UTILITY_ROOT = Path(os.environ.get(
    "MATRIX_PARTITION_UTILITY_HOME",
    REORDERING_ROOT / "Matrix-Partitioning-Utility",
))


def missing(message):
    print(f"missing_dependency: {message}", file=sys.stderr)
    return 42


def run(cmd, cwd=None, stdout=None, env=None):
    print(" ".join(str(x) for x in cmd), file=sys.stderr)
    return subprocess.run(cmd, cwd=cwd, stdout=stdout, stderr=subprocess.PIPE, text=True, env=env)


def ensure_output(path):
    if not Path(path).is_file() or Path(path).stat().st_size == 0:
        return False
    return True


def read_order_values(path):
    values = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = ""
        for line in f:
            stripped = line.strip()
            if stripped:
                first = stripped
                break
        if not first:
            raise ValueError(f"empty order file: {path}")

        if first.lower().startswith("%%matrixmarket"):
            dims_seen = False
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("%"):
                    continue
                if not dims_seen:
                    dims_seen = True
                    continue
                values.append(int(stripped.split()[0]))
        else:
            if not first.startswith("%"):
                values.append(int(first.split()[0]))
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("%"):
                    continue
                values.append(int(stripped.split()[0]))
    return values


def normalize_order_file(input_order, output_order, semantics):
    values = read_order_values(input_order)
    n = len(values)
    if n == 0:
        raise ValueError(f"empty order file: {input_order}")

    min_id = min(values)
    max_id = max(values)
    if min_id >= 0 and max_id < n:
        zero_based = values
    elif min_id >= 1 and max_id <= n:
        zero_based = [x - 1 for x in values]
    else:
        raise ValueError(
            f"order ids out of range in {input_order}: min={min_id}, max={max_id}, n={n}"
        )

    seen = [False] * n
    for idx, value in enumerate(zero_based):
        if value < 0 or value >= n:
            raise ValueError(f"invalid id {value} at line {idx + 1}")
        if seen[value]:
            raise ValueError(f"duplicate id {value} in {input_order}")
        seen[value] = True

    if semantics == "new_to_old":
        new_to_old = zero_based
    elif semantics == "old_to_new":
        new_to_old = [0] * n
        for old_id, new_id in enumerate(zero_based):
            new_to_old[new_id] = old_id
    else:
        raise ValueError(f"unknown order semantics: {semantics}")

    out = Path(output_order)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for old_id in new_to_old:
            f.write(f"{old_id}\n")


def sparsebase_order(kind, input_mtx, output_order):
    sparsebase_home = Path(os.environ.get("SPARSEBASE_HOME", REORDERING_ROOT / "SparseBase"))
    binary_by_kind = {
        "amd": sparsebase_home / "build/examples/amd_order/amd_order",
        "degree": sparsebase_home / "build/examples/degree_order/degree_order_mtx",
        "gray": sparsebase_home / "build/examples/gray_order/gray_order",
        "slashburn": sparsebase_home / "build/examples/slashburn_order/slashburn_order",
    }
    binary = binary_by_kind[kind]
    if not binary.exists():
        return missing(f"{kind} requires built SparseBase binary: {binary}")
    with tempfile.NamedTemporaryFile(prefix=f"{kind}_", suffix=".order") as raw:
        result = run([str(binary), input_mtx, raw.name, "1"])
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            return result.returncode
        try:
            normalize_order_file(raw.name, output_order, "old_to_new")
        except Exception as exc:
            print(f"{kind} order normalization failed: {exc}", file=sys.stderr)
            return 1
        raw_inv = Path(raw.name + "_inv")
        if raw_inv.exists():
            raw_inv.unlink()
    return 0 if ensure_output(output_order) else 1


def libmtx_order(kind, input_mtx, output_order):
    mtxreorder = shutil.which("mtxreorder")
    if mtxreorder is None:
        libmtx_home = Path(os.environ.get("LIBMTX_HOME", "/home/hangcheng.dong/PPoPP27/libmtx"))
        candidate = libmtx_home / "install/bin/mtxreorder"
        if candidate.exists():
            mtxreorder = str(candidate)
    if mtxreorder is None:
        return missing(f"{kind} requires mtxreorder from libmtx in PATH")
    ordering = {"nd": "nd", "rcm": "rcm"}[kind]
    with tempfile.NamedTemporaryFile(prefix=f"{kind}_", suffix=".mtx") as reordered, \
            tempfile.NamedTemporaryFile(prefix=f"{kind}_", suffix=".order") as raw_order:
        result = run([
            mtxreorder,
            "--verbose",
            input_mtx,
            f"--ordering={ordering}",
            f"--rowperm-path={raw_order.name}",
        ], stdout=reordered)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            return result.returncode
        try:
            normalize_order_file(raw_order.name, output_order, "old_to_new")
        except Exception as exc:
            print(f"{kind} order normalization failed: {exc}", file=sys.stderr)
            return 1
    return 0 if ensure_output(output_order) else 1


def rabbit_order(input_mtx, output_order):
    rabbit_home = Path(os.environ.get("RABBIT_HOME", REORDERING_ROOT / "rabbit_order"))
    converter = rabbit_home / "converter/mtx_to_el"
    reorder = rabbit_home / "demo/reorder"
    if not converter.exists():
        return missing(f"rabbit requires converter binary: {converter}")
    if not reorder.exists():
        return missing(f"rabbit requires demo reorder binary: {reorder}")
    with tempfile.TemporaryDirectory(prefix="rabbit_order_") as td:
        edge_list = Path(td) / "input.el"
        community = Path(td) / "input.off"
        result = run([str(converter), input_mtx, str(edge_list)])
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            return result.returncode
        reorder_env = os.environ.copy()
        reorder_env["OMP_NUM_THREADS"] = os.environ.get("TASA_RABBIT_THREADS", "1")
        result = run([str(reorder), str(edge_list), output_order, str(community), "-s"], env=reorder_env)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            return result.returncode
        raw_order = Path(td) / "rabbit.raw.order"
        Path(output_order).rename(raw_order)
        try:
            normalize_order_file(raw_order, output_order, "old_to_new")
        except Exception as exc:
            print(f"rabbit order normalization failed: {exc}", file=sys.stderr)
            return 1
    return 0 if ensure_output(output_order) else 1


def partition_order(kind, input_mtx, output_order):
    cache_dir = os.environ.get("TASA_PREGENERATED_ORDER_DIR")
    if cache_dir:
        candidate = Path(cache_dir) / kind / (Path(input_mtx).stem + f".{kind}order")
        if candidate.is_file():
            try:
                normalize_order_file(candidate, output_order, "new_to_old")
            except Exception as exc:
                print(f"{kind} pregenerated order normalization failed: {exc}", file=sys.stderr)
                return 1
            return 0
        print(f"{kind} pregenerated order not found: {candidate}; using AE partition workflow", file=sys.stderr)

    converter = PARTITION_UTILITY_ROOT / "Converter/PartitiontoReorderConverter"
    if not converter.exists():
        return missing(f"{kind} requires partition converter binary: {converter}")

    parts = os.environ.get("TASA_PARTITION_PARTS", "64")
    with tempfile.TemporaryDirectory(prefix=f"{kind}_partition_") as td:
        workdir = Path(td)
        (workdir / "output").mkdir()
        (workdir / "status").mkdir()

        stem = Path(input_mtx).stem
        if kind == "gp":
            utility_home = Path(os.environ.get(
                "GP_PARTITION_HOME",
                PARTITION_UTILITY_ROOT / "METIS",
            ))
            binary = utility_home / "run"
            if not binary.exists():
                return missing(f"gp requires METIS partition binary: {binary}")
            partvec = workdir / "output" / f"{stem}_metis_edge-cut_part{parts}.txt"
            result = run([
                str(binary),
                "-i", input_mtx,
                "-k", parts,
                "-o", "edge-cut",
                "-t",
                "-s",
            ], cwd=workdir)
        elif kind == "hp":
            utility_home = Path(os.environ.get(
                "HP_PARTITION_HOME",
                PARTITION_UTILITY_ROOT / "PaToH",
            ))
            binary = utility_home / "a.out"
            if not binary.exists():
                return missing(f"hp requires PaToH partition binary: {binary}")
            partvec = workdir / "output" / f"PaToH_{Path(input_mtx).name}_cutpart_k{parts}_quality_s1_partvec.txt"
            result = run([
                str(binary),
                input_mtx,
                "cutpart",
                parts,
                "quality",
                "1",
                "1",
            ], cwd=workdir)
        else:
            print(f"unknown partition reorder kind: {kind}", file=sys.stderr)
            return 2

        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            return result.returncode
        if not ensure_output(partvec):
            print(f"{kind} partition vector not produced: {partvec}", file=sys.stderr)
            return 1

        raw_order = workdir / f"{kind}.raw.order"
        result = run([str(converter), str(partvec), str(raw_order)], cwd=workdir)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            return result.returncode
        try:
            normalize_order_file(raw_order, output_order, "new_to_old")
        except Exception as exc:
            print(f"{kind} partition order normalization failed: {exc}", file=sys.stderr)
            return 1
    return 0 if ensure_output(output_order) else 1


def main(kind, argv):
    if len(argv) != 3:
        print(f"usage: {argv[0]} <input.mtx> <output.order>", file=sys.stderr)
        return 2
    input_mtx, output_order = argv[1], argv[2]
    if kind in {"amd", "degree", "gray", "slashburn"}:
        return sparsebase_order(kind, input_mtx, output_order)
    if kind in {"nd", "rcm"}:
        return libmtx_order(kind, input_mtx, output_order)
    if kind == "rabbit":
        return rabbit_order(input_mtx, output_order)
    if kind in {"gp", "hp"}:
        return partition_order(kind, input_mtx, output_order)
    print(f"unknown external reorder kind: {kind}", file=sys.stderr)
    return 2
