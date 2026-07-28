# Hugging Face dataset licensing (DroneBench)

The [anonymous DroneBench Hugging Face dataset](https://huggingface.co/datasets/anonymoussubmission2/anonymous-submission-neurips26-2831/tree/main) is **not** a pure MIT-licensed release. It is a composite benchmark derived from upstream sources with distinct licenses:

| Component | License |
|-----------|---------|
| Sim2Real-Fire simulation data | Apache-2.0 |
| USFS Burn Probability / Wildfire Hazard Potential (BP/WHP) | CC BY |
| FPA FOD ignition records | CC0 1.0 (public domain) |
| WFDroneBench code (GitHub repo) | MIT |

## Do not tag the dataset as `license: mit`

The Hugging Face repository metadata should **not** use `license: mit` for the dataset card or repository tags. MIT applies to the companion **software** in this GitHub repository only.

Recommended Hugging Face metadata:

- Set the dataset card license field to reflect **composite / multi-license** provenance (for example, list Apache-2.0, CC-BY-4.0, and CC0-1.0 in the card body).
- Include the root `NOTICE` (or equivalent attribution text) in every published archive.
- Link to this file and the root `NOTICE` for full attribution details.

## What users must do

Anyone redistributing DroneBench must:

1. Preserve upstream attribution for Sim2Real-Fire, USFS BP/WHP, and FPA FOD.
2. Comply with Apache-2.0, CC BY, and CC0 terms for the respective components.
3. Not represent the composite dataset as solely MIT-licensed.

See also `NOTICE` at the repository root for release packaging.
