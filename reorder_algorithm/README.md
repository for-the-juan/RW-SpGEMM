# Reorder Algorithm Adapters

Each subdirectory exposes a uniform adapter:

```bash
order_gen <input.mtx> <output.order>
```

`output.order` is a zero-based permutation with one integer per line, using
`new_id -> old_id` semantics. Adapters normalize external tools that natively
emit `old_id -> new_id`, so TileSpGEMM sees one format only.

TileSpGEMM uses these adapters as preprocessing when launched with
`-reorder 1 -reorder_algo <name>`:

- `AA`: one order from A is applied to row, inner, and column axes.
- `AAT`: one order from A is applied to A and A^T axes for square matrices.
- `AB`: one order from A is used for row/inner axes, and one order from B is
  used for the output column axis. The current external path is intended for
  the square same-dimension matrix pairs already used by the AB experiments.

Native adapters available without third-party builds:

- `original`

Adapters that require the original external toolchain:

- `amd`, `degree`, `gray`, `slashburn` require SparseBase build outputs.
- `rcm` and `nd` require `mtxreorder` from libmtx in `PATH` or
  `/home/hangcheng.dong/PPoPP27/libmtx/install/bin`.
- `rabbit` requires `rabbit_order/converter/mtx_to_el` and `rabbit_order/demo/reorder`.
  The wrapper runs Rabbit's reorder stage with `OMP_NUM_THREADS=1` by default
  for deterministic output; override with `TASA_RABBIT_THREADS`.
- `gp` follows the AE workflow: METIS 64-way edge-cut partitioning, then
  `Matrix-Partitioning-Utility/Converter/PartitiontoReorderConverter`.
- `hp` follows the AE workflow: PaToH 64-way cutpart quality partitioning,
  then `Matrix-Partitioning-Utility/Converter/PartitiontoReorderConverter`.

`gp` and `hp` can still reuse pregenerated orders when
`TASA_PREGENERATED_ORDER_DIR` is set and contains `<algo>/<matrix>.<algo>order`.
Otherwise, they generate the partition and order on the fly. Override the
partition count with `TASA_PARTITION_PARTS`; the AE default is `64`.

`random` remains as an unsafe diagnostic adapter only. It is not an AE reorder
scheme and is disabled by default because it failed SpGEMM correctness in the
kernel path; set `TASA_ENABLE_UNSAFE_RANDOM=1` only for isolated debugging.

Set `TASA_REORDER_ALGORITHM_ROOT` to this directory when running TileSpGEMM
from another working directory.

Batch entry point:

```bash
./run_reorder_algorithms_aa_aat_ab.sh
```

The script uses `/home/hangcheng.dong/PPoPP27/mat-142/mat-142` by default and
writes logs plus `summary.csv` under `/tmp/tasa_reorder_algorithms_logs`.
