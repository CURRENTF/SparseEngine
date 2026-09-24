# Decode-efficiency figures: 128K and 32K input, 2K output

Device: NVIDIA H100 80GB HBM3. Qwen3-30B-A3B-Instruct-2507-FP8 uses TP1; GLM-4.7-Flash (BF16) uses TP2/EP2. Both panels use `boundary_sync_v2`: 32 warmup decode steps and a continuous 256-step full-residency decode window, one discarded and three measured workloads. The plotted rate is pooled actual decode tokens divided by pooled window seconds, not serving throughput.

This export contains the exact portable data used for the six figures: `128k/plot_data.json`, `32k/plot_data.json`, and the absolute, relative-to-vLLM, and largest-measured-batch CSVs in each directory. The PNG, PDF, and SVG files remain in the persistent output directory `/data2/haojitai/outputs/Sparse-vLLM/paper_decode_figures_128k32k_20260923_final_b_equals/`, with `128k/absolute`, `128k/relative`, `32k/absolute`, and `32k/relative` subdirectories. Each group uses the normal low-batch line, relative-vLLM line, and largest-measured-batch bar; the extra log-y artifacts are not part of the six-figure set.

The 128K native Vanilla, SnapKV, QuEST, and OmniKV points are from the 2026-09-23 repository-optimization rerun. The unchanged vLLM, Tangram, HiSparse, and Vortex curves are reused from validated earlier `boundary_sync_v2` artifacts; they were not remeasured in this rerun. Qwen SnapKV has a confirmed B=30 boundary (B=31 failed capacity). GLM SnapKV has a clean B=55 measurement but no classified capacity upper bound. All bars use `B=N` to denote the largest successfully measured batch, including those without a confirmed capacity maximum; `capacity_confirmed` in the CSV preserves that distinction. GLM Tangram and HiSparse are unsupported and are absent. H2O is not included in this comparison.

The 32K figures use the latest 2026-09-23 presentation export, including the available external baselines. Its native results include the 2026-09-22 repository-optimization measurements. Bars show throughput at each method's largest validated batch, not each method's peak throughput. Within each model, Ours precedes the baselines and x-axis method names omit parentheses. On both input lengths, the Qwen line display stops at B4 and the GLM line display stops at B8; the portable data retain the measured larger batches used for bars. Relative lines use only exactly matched batch sizes against vLLM Vanilla; missing matches are omitted without interpolation.

Measurement commits recorded in the native artifacts: 128K `58138120c4cbb55dbce7f7fb7525bbbb64ac0b8d`; 32K `a07c3e30be63646137cc6aeeff131adc987afe5d` and `469aa791876380d0fc3dc850cfc083e973e157d4`. Source data and per-point artifact paths are retained in the portable JSON/CSV.

To render again with the current plotter, set `RESULT_DIR` to this directory and `FIGURE_DIR` to a new persistent output directory, then run:

```bash
for length in 128k 32k; do
  if [ "$length" = 128k ]; then input_len=131072; else input_len=32768; fi
  python scripts/official_experiments/sparse_decode_efficiency/plot_decode_capacity.py \
    --plot-data "$RESULT_DIR/$length/plot_data.json" --input-len "$input_len" \
    --output-dir "$FIGURE_DIR/$length/absolute"
  python scripts/official_experiments/sparse_decode_efficiency/plot_decode_capacity.py \
    --plot-data "$RESULT_DIR/$length/plot_data.json" --input-len "$input_len" \
    --line-metric relative-vllm --output-dir "$FIGURE_DIR/$length/relative"
done
```
