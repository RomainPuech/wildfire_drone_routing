# HuggingFace Upload Instructions for MasterYoda293

## Prerequisites

```bash
pip install huggingface_hub datasets pyarrow
huggingface-cli login   # paste your HF write token
```

## Step 1: Generate Parquet files

Run from the repository root:

```bash
python hf_release/build_parquet.py
```

This creates:
- `hf_release/data/scenarios_index.parquet` — full 7 746-scenario index
- `hf_release/data/tables23_scenarios.parquet` — 474-scenario Tables 2/3 split

## Step 2: Upload to HuggingFace

### Option A — CLI push (recommended)

```bash
cd hf_release

# Upload the dataset card and data files
huggingface-cli upload MasterYoda293/WFDroneBench README.md README.md --repo-type dataset
huggingface-cli upload MasterYoda293/WFDroneBench NOTICE NOTICE --repo-type dataset
huggingface-cli upload MasterYoda293/WFDroneBench data/ data/ --repo-type dataset
```

### Option B — Python API

```python
from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="hf_release/README.md",
    path_in_repo="README.md",
    repo_id="MasterYoda293/WFDroneBench",
    repo_type="dataset",
)
api.upload_file(
    path_or_fileobj="hf_release/NOTICE",
    path_in_repo="NOTICE",
    repo_type="dataset",
    repo_id="MasterYoda293/WFDroneBench",
)
api.upload_folder(
    folder_path="hf_release/data",
    path_in_repo="data",
    repo_id="MasterYoda293/WFDroneBench",
    repo_type="dataset",
)
```

## Step 3: Verify

```python
from datasets import load_dataset

# Should return a DatasetDict with a "train" split
ds = load_dataset("MasterYoda293/WFDroneBench", "default")
print(ds)
print(ds["train"][0])

# Should return a DatasetDict with a "test" split
ds_t23 = load_dataset("MasterYoda293/WFDroneBench", "tables23")
print(ds_t23)
print(ds_t23["test"][0])
```

## Step 4: Update repo metadata

1. On HuggingFace, go to Settings → change license from `mit` to `cc-by-4.0`
2. Verify the Dataset Viewer loads both configs correctly
3. Remove the old raw `DroneBench.zip` or keep it alongside the parquet files

## What changed from the original upload

| Before | After |
|--------|-------|
| Single `DroneBench.zip`, no loading script | Parquet index + 2 configs (`default`, `tables23`) |
| `license: mit` | `license: cc-by-4.0` (matches upstream CC-BY data) |
| No NOTICE / attribution | `NOTICE` with full source attribution |
| `load_dataset` returns raw zip | `load_dataset` returns structured `Dataset` |
| No Tables 2/3 split published | `tables23` config with 474 scenarios |
