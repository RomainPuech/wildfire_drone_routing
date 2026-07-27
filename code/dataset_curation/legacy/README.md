# Recovered historical artifacts

Recovery date: **2026-07-27**.

These files are preserved, unmodified provenance artifacts. They contain
hardcoded paths and destructive operations and must not be used as the
published curation entrypoint. No generated TIF, NPY, dataset content, Python
cache, or macOS metadata is included.

The same original WFDroneBench curation lineage is present in repository
history (`86f8062` through `dff00cf`) and on `main`; it was removed from the
public/R1 line in `96d93cb`. The last pre-removal tree is
`00281b8321adcc667360bb7f89f716e6045d6a0f` (`96d93cb^`). A second local
recovery copy exists under
`/Users/romain/Desktop/archive/Desktop/BKP/dataset_creation/`. These locations
were used only to cross-check provenance; they are not runtime dependencies.

Supercloud source paths:

- `/home/gridsan/groups/WFDroneBench/code.zip`
  - `code/dataset_creation/dataset_creation.ipynb`
  - `code/dataset_creation/whp_riskmap.ipynb`
  - `code/dataset_creation/Scenario_sampler.py`
  - `code/dataset_creation/generate_csv.py`
  - `code/dataset_creation/move_maps.py`
  - `code/dataset.py`
- `/home/gridsan/groups/WFDroneBench/preprocess.py`
- `/home/gridsan/groups/WFDroneBench/cleandataset.py`

The tracked repository-root `config_s2r.json` was also recovered from
`/home/gridsan/groups/WFDroneBench/config_s2r.json`, but is not duplicated
here because the mapping is already versioned at the repository root.

The supported implementation is the parent `dataset_curation` package. Its
README documents provenance gaps and every intentional deviation from these
artifacts.
