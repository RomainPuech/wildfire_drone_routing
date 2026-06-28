report_final bundle (California 2021 greedy-uniform, final_nature + April 13 context)
======================================================================================

Archive: **report_final_20260413_bundle.zip** (same folder as this file).

Contents of the zip
-------------------
report_final.md
  Main write-up (final_nature MaxCov/LinearMinTime + clustered TOP/MaxCov from April note).

benchmark_2021_greedy_kernel.md
  April 13, 2026 benchmark report (methodology, clustered routing tables).

figures/
  PNGs referenced by report_final.md (fire sample + 20M/100M placement maps; 500M maps included for completeness).

csv/
  benchmark_results_yearly_20260413_{162445,170201,175828,203043}.csv — final_nature yearly results (project-root copies).
  benchmark_results_yearly_20260412_085132.csv — 100M TOPGrowing slice (Apr bundle).
  benchmark_results_yearly_greedy_uniform_20260411_13_merged.csv — merged Apr 11–13 greedy-uniform rows.

supercloud_final_nature_routing_array.sh
  Slurm driver used for final_nature runs.

logs/
  wf_fnature_route-4498086_{0..3}.{out,err} — successful Slurm array job logs.

Placement JSONs (not in this zip; large / dataset-specific)
------------------------------------------------------------
California2021Dataset/logs/sensor_alloc_GaussianBudget{20,100}M_StationMaxGreedyUniform_261x161_mean.json

Unpack and open report_final.md next to the figures/ folder so relative image links resolve.
