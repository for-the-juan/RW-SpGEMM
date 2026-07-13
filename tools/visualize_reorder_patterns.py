#!/usr/bin/env python3

import argparse
import math
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


BLOCK_SIZE = 16
DEFAULT_DATASET_DIR = Path("/home/hangcheng.dong/PPoPP27/mat-142/mat-142")

DEFAULT_CASES = [
    {
        "id": "aa_rajat20",
        "title": "AA: rajat20",
        "kind": "AA",
        "aat": 0,
        "a": "rajat20.mtx",
        "b": None,
    },
    {
        "id": "ab_mult_dcop",
        "title": "AB: mult_dcop_01 x mult_dcop_02",
        "kind": "AB",
        "aat": 0,
        "a": "mult_dcop_01.mtx",
        "b": "mult_dcop_02.mtx",
    },
    {
        "id": "ab_rajat",
        "title": "AB: rajat17 x rajat16",
        "kind": "AB",
        "aat": 0,
        "a": "rajat17.mtx",
        "b": "rajat16.mtx",
    },
]

BLUE_CMAP = LinearSegmentedColormap.from_list(
    "white_to_deep_blue",
    ["#ffffff", "#e8f3ff", "#badcff", "#6aaee8", "#2171b5", "#08306b"],
)


def read_mtx_coords(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        banner = f.readline().strip().split()
        if len(banner) < 5 or banner[0].lower() != "%%matrixmarket":
            raise ValueError(f"{path} is not a MatrixMarket coordinate file")
        symmetry = banner[4].lower()
        symmetric = symmetry in ("symmetric", "hermitian")

        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        m, n, reported_nnz = [int(x) for x in line.split()[:3]]

        rows = np.empty(reported_nnz * (2 if symmetric else 1), dtype=np.int64)
        cols = np.empty_like(rows)
        count = 0
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            r = int(parts[0]) - 1
            c = int(parts[1]) - 1
            rows[count] = r
            cols[count] = c
            count += 1
            if symmetric and r != c:
                rows[count] = c
                cols[count] = r
                count += 1

    return rows[:count], cols[:count], m, n


def read_perm(path):
    data = np.loadtxt(path, dtype=np.int64)
    if data.ndim == 0:
        return np.empty(0, dtype=np.int64)
    n = int(data[0])
    perm = data[1:]
    if len(perm) != n:
        raise ValueError(f"bad permutation length in {path}: expected {n}, got {len(perm)}")
    return perm


def read_meta(path):
    meta = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "=" not in line:
                continue
            key, value = line.strip().split("=", 1)
            meta[key] = value
    return meta


def run_dump(dump_bin, dataset_dir, out_dir, case):
    prefix = out_dir / "perms" / case["id"]
    prefix.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(dump_bin),
        "-aat",
        str(case["aat"]),
        str(dataset_dir / case["a"]),
        "-o",
        str(prefix),
    ]
    if case["b"]:
        cmd.extend(["-b", str(dataset_dir / case["b"])])
    subprocess.run(cmd, check=True)
    return prefix


def binned_count(rows, cols, m, n, bins):
    ybins = min(bins, max(m, 1))
    xbins = min(bins, max(n, 1))
    y = (rows * ybins // max(m, 1)).astype(np.int64)
    x = (cols * xbins // max(n, 1)).astype(np.int64)
    y = np.clip(y, 0, ybins - 1)
    x = np.clip(x, 0, xbins - 1)
    heat = np.zeros((ybins, xbins), dtype=np.float32)
    np.add.at(heat, (y, x), 1.0)
    return heat


def log_image(heat):
    image = np.log1p(heat)
    positive = image[image > 0]
    if positive.size == 0:
        return image, 1.0
    vmax = float(np.percentile(positive, 99.5))
    return image, max(vmax, 1.0)


def draw_heatmap(ax, rows, cols, m, n, bins, title, xlabel):
    heat = binned_count(rows, cols, m, n, bins)
    image, vmax = log_image(heat)
    ax.imshow(image, cmap=BLUE_CMAP, origin="upper", interpolation="nearest", aspect="auto", vmin=0.0, vmax=vmax)
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=9)
    for spine in ax.spines.values():
        spine.set_color("#d9e6f2")
        spine.set_linewidth(0.6)


def tile_pairs(rows, cols, m, n):
    tile_m = (m + BLOCK_SIZE - 1) // BLOCK_SIZE
    tile_n = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    tr = rows // BLOCK_SIZE
    tc = cols // BLOCK_SIZE
    packed = tr * tile_n + tc
    packed = np.unique(packed)
    return packed // tile_n, packed % tile_n, tile_m, tile_n


def symbolic_c_tiles(a_tile_r, a_tile_c, b_tile_r, b_tile_c, c_tile_n):
    if len(a_tile_r) == 0 or len(b_tile_r) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    b_order = np.argsort(b_tile_r, kind="mergesort")
    b_k = b_tile_r[b_order]
    b_j = b_tile_c[b_order]
    b_map = {}
    start = 0
    while start < len(b_k):
        end = start + 1
        while end < len(b_k) and b_k[end] == b_k[start]:
            end += 1
        b_map[int(b_k[start])] = np.unique(b_j[start:end])
        start = end

    a_order = np.argsort(a_tile_c, kind="mergesort")
    a_k = a_tile_c[a_order]
    a_i = a_tile_r[a_order]
    chunks = []
    start = 0
    max_pairs_per_chunk = 2_000_000
    while start < len(a_k):
        end = start + 1
        while end < len(a_k) and a_k[end] == a_k[start]:
            end += 1
        js = b_map.get(int(a_k[start]))
        if js is not None and len(js) > 0:
            is_ = np.unique(a_i[start:end])
            i_chunk = max(1, max_pairs_per_chunk // len(js))
            for offset in range(0, len(is_), i_chunk):
                is_part = is_[offset:offset + i_chunk]
                packed = np.repeat(is_part, len(js)) * c_tile_n + np.tile(js, len(is_part))
                chunks.append(packed.astype(np.int64, copy=False))
        start = end

    if not chunks:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    packed = np.unique(np.concatenate(chunks))
    return packed // c_tile_n, packed % c_tile_n


def tile_count_stats(tile_rows):
    return {"tiles": int(len(tile_rows))}


def reduction_text(before, after):
    if before <= 0:
        return "n/a"
    reduction = 100.0 * (1.0 - float(after) / float(before))
    return f"{reduction:.1f}% fewer"


def input_metric_line(meta):
    return (
        f"mode={meta.get('mode', '')} | reason={meta.get('reason', '')} | "
        f"A tiles {meta.get('input_tiles_A_before', '?')} -> {meta.get('input_tiles_A_after', '?')} | "
        f"B tiles {meta.get('input_tiles_B_before', '?')} -> {meta.get('input_tiles_B_after', '?')} | "
        f"est. C tiles {meta.get('estimated_c_tiles_before', '?')} -> {meta.get('estimated_c_tiles_after', '?')}"
    )


def save_input_figure(out_dir, case, meta, matrices, bins):
    if case["kind"] == "AA":
        rows, cols, rows_after, cols_after, m, n = matrices["A"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True, facecolor="white")
        draw_heatmap(axes[0], rows, cols, m, n, bins, "A before", f"{m} x {n}, nnz={len(rows):,}")
        draw_heatmap(axes[1], rows_after, cols_after, m, n, bins, "A after row/inner reorder", f"{m} x {n}, nnz={len(rows):,}")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10.5), constrained_layout=True, facecolor="white")
        a_rows, a_cols, a_rows_after, a_cols_after, a_m, a_n = matrices["A"]
        b_rows, b_cols, b_rows_after, b_cols_after, b_m, b_n = matrices["B"]
        draw_heatmap(axes[0, 0], a_rows, a_cols, a_m, a_n, bins, "A before", f"{a_m} x {a_n}, nnz={len(a_rows):,}")
        draw_heatmap(axes[0, 1], a_rows_after, a_cols_after, a_m, a_n, bins, "A after row/inner reorder", f"{a_m} x {a_n}, nnz={len(a_rows):,}")
        draw_heatmap(axes[1, 0], b_rows, b_cols, b_m, b_n, bins, "B before", f"{b_m} x {b_n}, nnz={len(b_rows):,}")
        draw_heatmap(axes[1, 1], b_rows_after, b_cols_after, b_m, b_n, bins, "B after inner/col reorder", f"{b_m} x {b_n}, nnz={len(b_rows):,}")

    fig.suptitle(f"{case['title']} input nonzero density\n{input_metric_line(meta)}", fontsize=12)
    out_png = out_dir / f"{case['id']}_inputs_blue.png"
    fig.savefig(out_png, dpi=240, facecolor="white")
    plt.close(fig)
    return out_png


def save_c_tile_figure(out_dir, case, meta, c_before, c_after, c_tile_m, c_tile_n, bins):
    before_stats = tile_count_stats(c_before[0])
    after_stats = tile_count_stats(c_after[0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True, facecolor="white")
    draw_heatmap(
        axes[0],
        c_before[0],
        c_before[1],
        c_tile_m,
        c_tile_n,
        bins,
        "symbolic C tiles before",
        f"occupied 16x16 tiles={before_stats['tiles']:,}",
    )
    draw_heatmap(
        axes[1],
        c_after[0],
        c_after[1],
        c_tile_m,
        c_tile_n,
        bins,
        "symbolic C tiles after",
        f"occupied 16x16 tiles={after_stats['tiles']:,}",
    )
    fig.suptitle(
        f"{case['title']} product tile evidence\n"
        f"computed C tiles {before_stats['tiles']:,} -> {after_stats['tiles']:,} "
        f"({reduction_text(before_stats['tiles'], after_stats['tiles'])}); "
        f"C++ estimate {meta.get('estimated_c_tiles_before', '?')} -> {meta.get('estimated_c_tiles_after', '?')}",
        fontsize=12,
    )
    out_png = out_dir / f"{case['id']}_c_tiles_blue.png"
    fig.savefig(out_png, dpi=240, facecolor="white")
    plt.close(fig)

    evidence = {
        "c_before": before_stats,
        "c_after": after_stats,
    }
    return out_png, evidence


def visualize_case(dataset_dir, out_dir, dump_bin, case, bins, tile_bins):
    prefix = run_dump(dump_bin, dataset_dir, out_dir, case)
    row_perm = read_perm(str(prefix) + "_row_old_to_new.txt")
    inner_perm = read_perm(str(prefix) + "_inner_old_to_new.txt")
    col_perm = read_perm(str(prefix) + "_col_old_to_new.txt")
    meta = read_meta(str(prefix) + "_meta.txt")

    a_rows, a_cols, a_m, a_n = read_mtx_coords(dataset_dir / case["a"])
    a_rows_after = row_perm[a_rows]
    a_cols_after = inner_perm[a_cols]

    if case["kind"] == "AA":
        b_rows, b_cols, b_m, b_n = a_rows, a_cols, a_m, a_n
        b_rows_after = inner_perm[b_rows]
        b_cols_after = col_perm[b_cols]
        matrices = {"A": (a_rows, a_cols, a_rows_after, a_cols_after, a_m, a_n)}
    else:
        b_rows, b_cols, b_m, b_n = read_mtx_coords(dataset_dir / case["b"])
        b_rows_after = inner_perm[b_rows]
        b_cols_after = col_perm[b_cols]
        matrices = {
            "A": (a_rows, a_cols, a_rows_after, a_cols_after, a_m, a_n),
            "B": (b_rows, b_cols, b_rows_after, b_cols_after, b_m, b_n),
        }

    a_tr, a_tc, a_tm, a_tn = tile_pairs(a_rows, a_cols, a_m, a_n)
    b_tr, b_tc, b_tm, b_tn = tile_pairs(b_rows, b_cols, b_m, b_n)
    a_tr_after, a_tc_after, _, _ = tile_pairs(a_rows_after, a_cols_after, a_m, a_n)
    b_tr_after, b_tc_after, _, _ = tile_pairs(b_rows_after, b_cols_after, b_m, b_n)

    c_before = symbolic_c_tiles(a_tr, a_tc, b_tr, b_tc, b_tn)
    c_after = symbolic_c_tiles(a_tr_after, a_tc_after, b_tr_after, b_tc_after, b_tn)

    input_png = save_input_figure(out_dir, case, meta, matrices, bins)
    c_png, evidence = save_c_tile_figure(out_dir, case, meta, c_before, c_after, a_tm, b_tn, tile_bins)
    return {"case": case, "input_png": input_png, "c_png": c_png, "meta": meta, "evidence": evidence}


def fmt_ratio(before, after):
    if before == 0:
        return "n/a"
    return f"{after / before:.3f}x"


def write_report(out_dir, results):
    report = out_dir / "README.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write("# Reorder Pattern Visualizations\n\n")
        f.write("All figures use a white background and blue foreground. Pixel intensity is log-scaled density. The after panels apply the exact permutations dumped by the current C++ reorder plan.\n\n")
        f.write("The `*_inputs_blue.png` figures show input nonzero density. The `*_c_tiles_blue.png` figures show symbolic output tile occupancy, which is closer to the TileSpGEMM objective.\n\n")
        f.write("| case | mode | est. C tiles | computed C tiles | reduction | A input tiles | B input tiles |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for result in results:
            case = result["case"]
            meta = result["meta"]
            ev = result["evidence"]
            cb = ev["c_before"]
            ca = ev["c_after"]
            f.write(
                f"| {case['title']} | `{meta.get('mode', '')}` | "
                f"{meta.get('estimated_c_tiles_before', '?')} -> {meta.get('estimated_c_tiles_after', '?')} | "
                f"{cb['tiles']:,} -> {ca['tiles']:,} ({fmt_ratio(cb['tiles'], ca['tiles'])}) | "
                f"{reduction_text(cb['tiles'], ca['tiles'])} | "
                f"{meta.get('input_tiles_A_before', '?')} -> {meta.get('input_tiles_A_after', '?')} | "
                f"{meta.get('input_tiles_B_before', '?')} -> {meta.get('input_tiles_B_after', '?')} |\n"
            )
        f.write("\n")
        for result in results:
            case = result["case"]
            meta = result["meta"]
            f.write(f"## {case['title']}\n\n")
            f.write(f"- mode: `{meta.get('mode', '')}`\n")
            f.write(f"- reason: `{meta.get('reason', '')}`\n")
            f.write(f"- A input tiles: `{meta.get('input_tiles_A_before', '?')}` -> `{meta.get('input_tiles_A_after', '?')}`\n")
            f.write(f"- B input tiles: `{meta.get('input_tiles_B_before', '?')}` -> `{meta.get('input_tiles_B_after', '?')}`\n")
            f.write(f"- estimated C tiles: `{meta.get('estimated_c_tiles_before', '?')}` -> `{meta.get('estimated_c_tiles_after', '?')}`\n\n")
            f.write(f"![{case['id']} inputs]({result['input_png'].name})\n\n")
            f.write(f"![{case['id']} C tiles]({result['c_png'].name})\n\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("visualizations/reorder_patterns"))
    parser.add_argument("--dump-bin", type=Path, default=Path("/tmp/dump_reorder_plan"))
    parser.add_argument("--bins", type=int, default=768)
    parser.add_argument("--tile-bins", type=int, default=384)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for case in DEFAULT_CASES:
        result = visualize_case(args.dataset_dir, args.out_dir, args.dump_bin, case, args.bins, args.tile_bins)
        results.append(result)
        ev = result["evidence"]
        print(
            f"{case['id']}: inputs={result['input_png']} c_tiles={result['c_png']} "
            f"mode={result['meta'].get('mode')} C_tiles={ev['c_before']['tiles']}->{ev['c_after']['tiles']} "
            f"reduction={reduction_text(ev['c_before']['tiles'], ev['c_after']['tiles'])}"
        )

    report = write_report(args.out_dir, results)
    print(f"report: {report}")


if __name__ == "__main__":
    main()
