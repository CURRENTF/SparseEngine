# Palu author Hugging Face LongBench comparison

This run evaluates Llama-3.1-8B-Instruct with the Palu authors' Hugging Face
`PaluLlamaForCausalLM` implementation at upstream commit
`bb22666e2ef96707e8dd21d93fc00146c2e0d615`. It is distinct from the
[Hugging Face FullKV comparison](../palu_llama31_longbench_hf_fullkv_20260925/RESULTS.md).

The authors' model reconstructs K/V from low-rank projection modules before
standard Hugging Face attention, which stores full K/V in its cache. This run
therefore compares **Palu model quality**, not compressed-cache memory or speed.
The upstream standalone Palu attention kernel is not wired into this model.

The source patch in `palu_compat.patch` adds Llama 3.1 RoPE validation and
defers an unused Hadamard import. The conversion script maps the exact frozen
SparseEngine group-size-1, rank-96 K/V factors into the authors' per-head
`VT` and `U` modules, after checking all original K/V weight fingerprints.
The original factorization used activation whitening. All layers use K/V rank
96. The source model and factors are BF16.

`run_author_hf.py` uses the authors' model class with the existing frozen
LongBench v1/v2 prompt-token inputs and baseline scorer. Both evaluations use
greedy decoding, the same per-sample output budgets and EOS policy, and the
same input split and scoring definitions as the preceding SparseEngine Palu run.
The run applies last-token prefill logits and 8192-token MLP chunks to bound
temporary memory, without changing model weights or attention outputs.

Remote raw outputs, per-sample scores, logs, and converted checkpoint live at
`/root/autodl-fs/outputs/Sparse-vLLM/palu-llama31-author-hf-20260925` and
`/root/autodl-fs/checkpoints/palu/llama31_8b_whitened_g1_r96_author_hf_20260925`.
The validated scores are in `results.json` and summarized in `RESULTS.md`.
The SparseEngine Palu v2 category aggregate is preserved in
`sparseengine_palu_longbench_v2_metrics.json`.
Both scorers report success with 3,750 v1 samples and 503 v2 samples and no
model failures. The v2 scorer records 38 answer parse failures under its
unchanged metric policy.
