# Data

This directory contains the initial workbook, B1/B2 metrics, fixed-Q error
summary, candidate summaries, and manuscript Table 1.

## Files and sources

- `initial_samples/examples2.xlsx` contains ten initial samples. Composition
  columns are ordered `[Pd, Pt, Cu, Au, Ir, Ce, Nb, Cr]`; the response column
  is `max_power`.
- `benchmark_metrics/metrics_per_run.csv` contains 100 B1/B2 run-level rows
  (30 B1 and 70 B2); `metrics_summary.csv` contains 20 grouped rows (6 B1 and
  14 B2). The `file` column records the source path, and `run` records the run
  number; the referenced trajectory files are not included.
- `fixed_q_errors/fixed_q_error_summary.csv` reports MAE and RMSE for 20 fixed
  test compositions at `k_added` checkpoints 0, 3, 6, 12, 24, 45, and 90.
  Each value is summarized by mean and sample SD across five repeats;
  `history_size` equals 10 plus `k_added`.
- `candidate_summary/max_power_cost_62_335.json` and
  `candidate_summary/closest_to_75_073.json` contain saved predictions, costs,
  and full-precision compositions.
- `table1.csv` contains the ten manuscript Table 1 rows. Composition columns
  follow `[Pd, Pt, Cu, Au, Ir, Ce, Nb, Cr]`; the manuscript reports three
  decimal places.
