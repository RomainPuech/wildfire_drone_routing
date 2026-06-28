California 2021 greedy-uniform kernel benchmark -- share bundle (2026-04-13)
================================================================================

Contents:
  benchmark_2021_greedy_kernel.md          Main report (clustered routing, Apr 2026 refresh)
  benchmark_2021_greedy_kernel.pdf         PDF render of the same
  benchmark_2021_greedy_kernel_20260327_no_clustering.md
                                           Archived March report (no-clustering baseline)
  benchmark_results_yearly_greedy_uniform_20260411_13_merged.csv
                                           798 rows: all strategies/budgets from Apr 12-13 runs
                                           Columns source_csv / source_run_tag identify origin file
  source_csv_slices/                       Original per-run CSV exports (same as merged slices)
  figures/                                 PNG maps regenerated from current California2021Dataset + greedy-uniform JSONs (see report Figures section)
  make_benchmark_fire_locations_figure.py  Regenerates benchmark_fire_locations_budget_2021.png (displays.plot_fire_locations)
  print_placement_detectability.py         Prints placement-only detectability for benchmark n=100 (20M/100M/500M)
  parse_routing_mip_gaps.py                Summarizes Gurobi gap lines from Slurm routing logs (MaxCov / LinearMinTime)

Original timestamps in filenames (also under project root if you unpack elsewhere):
  benchmark_results_yearly_20260412_084333.csv  20M MaxCov
  benchmark_results_yearly_20260412_081018.csv  20M TOPGrowing
  benchmark_results_yearly_20260413_134640.csv  20M LinearMinTime
  benchmark_results_yearly_20260412_094256.csv  100M MaxCov
  benchmark_results_yearly_20260412_085132.csv  100M TOPGrowing (98 rows; 2 scenarios missing)
  benchmark_results_yearly_20260412_075513.csv  500M MaxCov
  benchmark_results_yearly_20260412_075505.csv  500M TOPGrowing
  benchmark_results_yearly_20260413_121953.csv  500M LinearMinTime

Not included: 100M LinearMinTime (job was still running when this bundle was built).

Rebuild PDF from markdown:
  cd report/benchmark_2021_greedy_kernel
  pandoc benchmark_2021_greedy_kernel.md -o benchmark_2021_greedy_kernel.pdf --pdf-engine=pdflatex -V geometry:margin=1in
