# Official experiment packages

Each experiment has a self-contained directory for its orchestration scripts,
parameter JSON, plotting code, and lightweight result exports. Drivers call the
canonical benchmark entrypoints; they do not implement a separate timing engine.
Supply machine-specific model, environment, checkout, and output paths at runtime.
Large logs, token outputs, and checkpoints remain outside Git.

- [128K input / 2K output decode capacity](sparse_decode_efficiency/README.md):
  two models, five methods, exact concurrency boundaries, linear/log-y figures.
- [Sparse-Engine vs Vortex](sparseengine_vs_vortex/README.md): guarded single-card
  QuEST and H2O-like comparisons, full-residency decode timing, configs,
  recorded JSON data and plots.
- [AIME 2024](aime/README.md): pass@1 across all 11 methods without auxiliary
  checkpoints, using shared sampling and engine settings adapted from MiniSWE.

Here “official” identifies the maintained experiment recipe, not universal
performance guarantees or parity with an upstream method implementation.
Recorded results retain Git commit/dirty status and caveats. Recipes should not
add source snapshots, archives, patches, per-file source hashes, or source-equality
gates unless explicitly requested. Keep configuration and measurement-data checks.
