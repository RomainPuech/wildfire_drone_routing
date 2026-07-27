# WFDroneBench curation

The supported, non-destructive curation implementation and complete command
sequence are in
[`code/dataset_curation/README.md`](../code/dataset_curation/README.md).

It covers:

1. inventorying and filtering the exact 30 m Sim2Real-Fire layout grids;
2. merging and deduplicating FPA-FOD and newer USFS ignition records;
3. space-only and date-aware fire-to-scenario matching without scenario reuse;
4. geospatial reprojection of continental BP/WHP products onto each layout;
5. canonical JPG-to-NPY conversion and empirical burn-map construction; and
6. generation of auditable scenario metadata.

The raw physical fire simulations are upstream Sim2Real-Fire outputs; this
repository curates rather than regenerates them. Original research
notebooks/scripts are preserved under `code/dataset_curation/legacy/`. The
supported implementation fixes their row/column, nondeterminism, destructive
file handling, and raster-alignment defects, so corrected selections may not
be ID-identical to the historical release.

The exact 12-layout/474-scenario paper run list is the authoritative recovered
experiment manifest under [`splits/`](../splits/), not a claimed bit-for-bit
output of the corrected pipeline. See
[`splits/SELECTION_RULE.md`](../splits/SELECTION_RULE.md) for details.
