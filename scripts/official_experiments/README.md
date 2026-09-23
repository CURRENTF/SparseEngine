# Official experiment packages

Each experiment has a self-contained directory for its orchestration scripts,
parameter JSON, plotting code, and official result package. Drivers call the
canonical benchmark entrypoints; they do not implement a separate timing engine.
Supply machine-specific model, environment, checkout, and output paths at runtime.

A completed official experiment must commit the data used for its reported tables
and figures in its own directory. Keep the result package compact: record the
device, launch arguments, final results, and Git commit. Do not add working-tree
status, patches, source snapshots, source archives, per-file hashes, staging state,
or source-recovery material. Large logs, token outputs, checkpoints, and other raw
evidence remain on persistent storage outside Git; they are not a substitute for
the committed official results.

- [Radix prefix pruning by conversation region](radix_prefix_prune/README.md):
  region annotations, shared-budget multi-range pruning, and recorded MiniSWE
  full300 results.
- [GLM-4.7-Flash conservative concurrency](glm47_conservative_concurrency/README.md):
  H100 BF16 and PRO6000 FP8, per-context operating points and recorded results.
- [GLM-4.7-Flash MiniSWE](chain_cache_miniswe/README.md): closed-loop
  SWE-bench Lite recipe and recorded five-method official results.
- [128K input / 2K output decode capacity](sparse_decode_efficiency/README.md):
  two models, five methods, exact concurrency boundaries, linear/log-y figures.
- [AIME 2024](aime/README.md): pass@1 across all 11 methods without auxiliary
  checkpoints, using shared sampling and engine settings adapted from MiniSWE.

Here “official” identifies the maintained experiment recipe, not universal
performance guarantees or parity with an upstream method implementation.
Research-Vault may retain private run history, failures, or raw-artifact indexes,
but it must not be the only copy of an official result or its table/plot inputs.
