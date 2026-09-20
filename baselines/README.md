# Baseline Runtime Sources

This directory keeps only the baseline code needed by SparseEngine's runtime
adapters. Standalone benchmark, evaluation, profiling, visualization, and
bundled benchmark-data implementations from vendored baselines are omitted.

Use the repository-owned entrypoints under `benchmark/` and the runbooks under
`docs/en/benchmarking/` for reproducible comparisons. The upstream README files
are retained for provenance and may still describe benchmark files that are not
part of this trimmed copy.

`DivPrune`, `FastVID`, `PACT`, and `VisionZip` are pinned upstream submodules;
their contents are not vendored into this repository and are therefore not
selectively trimmed here.
