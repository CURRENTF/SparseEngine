# Benchmark

本页汇总支持的 benchmark 入口及其规范运行方式。benchmark 代码的 source-tree map 参见 [`benchmark/README.md`](../../../benchmark/README.md)。

## Benchmark 目录

仓库当前覆盖多模态之外的多种 benchmark。选择 runner 时使用下表进行路由；实际结果声明必须关联每次 run 生成的 output artifact。

| 领域 | 入口 | 范围与说明 |
| --- | --- | --- |
| 效率与吞吐 | [`efficiency.md`](efficiency.md) | 匹配的 Sparse-Engine/vLLM length、并发度、churn、TP、延迟、吞吐和 GPU 活动率直接采样套件。 |
| Sparse-Engine microbenchmark | `benchmark/microbench.py` | Synthetic prompt length 下的 engine throughput、TTFT、prefill/decode throughput、ITL 和 GPU memory。 |
| 模拟 Deep Research | [`simulated-deep-research.md`](simulated-deep-research.md) | 通过 non-uniform smart router 运行 synthetic 10-round main-agent/subagent serving workload。 |
| Max-batch throughput | `scripts/benchmarks/run_sparseengine_max_batch_throughput.py` | 面向 capacity 的 Sparse-Engine stress/throughput run。 |
| LongBench | `benchmark/long_bench/pred.py`, `benchmark/long_bench/eval.py` | 原生 Sparse-Engine runner；使用 `--task` 选择子集，省略则运行完整 suite。 |
| MathBench, AIME, MATH-500 | `benchmark/math_bench/pred.py`, `benchmark/math_bench/eval.py` | 原生 Sparse-Engine runner，task 包括 `gsm8k`、`aime2024`、`math500` 和 `hmmt_nov`。 |
| SCBench | `benchmark/scbench/run_scbench.py`, `benchmark/scbench/run_scbench_preprocessed.py`, `benchmark/scbench/compute_scores.py`, `benchmark/scbench/run_kvzip_preprocessed.py` | 仓库自有 run 与 upstream SCBench baseline 的 shared-context benchmark 路径。 |
| Claw-Eval | `benchmark/claw_eval/run_sparseengine_claw_eval.sh` | 通过共享 Sparse-Engine OpenAI-compatible server 驱动外部 Claw-Eval checkout。 |
| SWE-bench Lite | [`swe-bench-lite.md`](swe-bench-lite.md) | 外部 mini-SWE-agent generation 加官方 SWE-bench Docker harness。 |
| 多模态 | [`multimodal/README.md`](multimodal/README.md) | Video QA 和 image QA runner，以及当前方法支持限制。 |
| RULER core | `benchmark/ruler_vt/pred.py` | 使用原生 benchmark adapter、自包含地评测 NIAH retrieval、variable tracking、CWE 和 FWE。 |
| NIAH | `benchmark/niah/test_niah.py` | 原生 Sparse-Engine needle-in-a-haystack 长上下文 runner。 |
| Regression harness | [`sparseengine-regression-tests.md`](sparseengine-regression-tests.md) | 固定的 LongBench/RULER quality、performance 和 stress 检查。 |

## 吞吐量 Benchmark

使用 `benchmark/microbench.py` 测量不同 synthetic prompt length 下的 TTFT、prefill throughput、decode throughput、ITL 和 GPU memory。

说明：

- 优先通过 `--hyper_params` 以 JSON object 传入 Sparse-Engine 设置。
- `--hyper_params` 接受 `sparse_method`、`engine_prefill_chunk_size` 和 `decode_keep_tokens` 等规范 runtime name。
- `--lengths` 表示 prompt length；脚本内部设置 `max_model_len = length + output_len + 100`。

Baseline：

```bash
python benchmark/microbench.py \
  --model_path <PATH_TO_BASE_MODEL> \
  --lengths 512000 \
  --batch_sizes 2 \
  --methods vanilla \
  --hyper_params '{"gpu_memory_utilization": 0.9}' \
  --output_dir benchmark/results/microbench/vanilla_512k
```

旧路径 `scripts/benchmarks/bench_sparse_engine.py` 继续作为已有 runbook 和 regression harness 的 compatibility wrapper。

## 使用 Sparse-Engine 运行 MathBench

以下示例可直接使用 Sparse-Engine engine 比较 GSM8K、AIME 2024、MATH-500 和 HMMT-style 任务。dataset 细节参见 [`benchmark/math_bench/README.md`](../../../benchmark/math_bench/README.md)。

Full-attention baseline：

```bash
python benchmark/math_bench/pred.py \
  --model qwen7b-full \
  --model_path <MODEL_ROOT>/DeepSeek-R1-Distill-Qwen-7B \
  --tokenizer_path <MODEL_ROOT>/DeepSeek-R1-Distill-Qwen-7B \
  --ws 1 \
  --batch_size 30 \
  --task aime2024 \
  --temperature 0.6 \
  --hyper_param '{"engine_prefill_chunk_size": 4096, "sparse_method": "vanilla"}'
```

OmniKV：

```bash
python benchmark/math_bench/pred.py \
  --model qwen7b-omnikv \
  --model_path <MODEL_ROOT>/DeepSeek-R1-Distill-Qwen-7B \
  --tokenizer_path <MODEL_ROOT>/DeepSeek-R1-Distill-Qwen-7B \
  --ws 1 \
  --batch_size 30 \
  --task aime2024 \
  --temperature 0.6 \
  --hyper_param '{"engine_prefill_chunk_size": 4096, "sparse_method": "omnikv", "full_attention_layers": "0,1,2,4,7,14", "decode_keep_tokens": 1024}'
```

DeltaKV 需要针对同一 base model 训练的 compressor。请将下面的 checkpoint path 替换为与实际模型匹配的 compressor。

```bash
python benchmark/math_bench/pred.py \
  --model qwen7b-deltakv \
  --model_path <MODEL_ROOT>/DeepSeek-R1-Distill-Qwen-7B \
  --tokenizer_path <MODEL_ROOT>/DeepSeek-R1-Distill-Qwen-7B \
  --ws 1 \
  --batch_size 30 \
  --task aime2024 \
  --temperature 0.6 \
  --hyper_param '{"engine_prefill_chunk_size": 512, "max_num_batched_tokens": 8192, "max_num_seqs_in_batch": 30, "sparse_method": "deltakv", "full_attention_layers": "0,1,2,4,7,14", "decode_keep_tokens": 1024, "deltakv_checkpoint_path": "<CHECKPOINT_ROOT>/<MATCHING_COMPRESSOR_DIR>", "deltakv_latent_dim": 256, "deltakv_latent_quant_bits": 4, "full_layer_kv_quant_bits": 4, "enable_full_layer_kivi_quant": true}'
```

通过 `sparse_method` 选择方法，通过 `deltakv_checkpoint_path` 传入 checkpoint。

一次 run 包含多个数学任务时，向 `--task` 传入逗号分隔值，例如 `gsm8k,aime2024,math500`。`--num_samples` 仅用于 smoke run，不要把仅 smoke 的行混入最终 benchmark 表。

## 使用 Sparse-Engine 运行 LongBench

此入口通过原生 Sparse-Engine engine 运行 LongBench：

```bash
python benchmark/long_bench/pred.py \
  --model qwen7b-omnikv \
  --model_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --tokenizer_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --ws 1 \
  --batch_size 1 \
  --task qasper,hotpotqa,multi_news \
  --hyper_param '{"engine_prefill_chunk_size": 4096, "sparse_method": "omnikv", "decode_keep_tokens": 2048, "full_attention_layers": "0,1,2,4,7,14", "recent_keep_tokens": 128, "sink_keep_tokens": 8}'
```

完整 LongBench run 省略 `--task`。切换到 DeltaKV 时，设置 `sparse_method="deltakv"` 并提供匹配的 `deltakv_checkpoint_path`。旧 manifest 中的 legacy `deltakv-less-memory*` method ID 会规范到同一个 runtime。

对于线性 chain prefix-cache trace，在
`scripts/benchmarks/bench_prefix_cache.py` 中选择 `chain_snapkv`、
`chain_h2o`、`chain_pyramidkv`、`chain_rkv` 或 `chain_skipkv`。Chain case 要求使用
`--workloads multiturn --history_update generated`：turn 0 记录服务端创建的
`chain_id`，后续 turn 复用该 ID；每条记录包含 chain 状态、复用的逻辑 token
数、新 prefill 的增量 token 数以及逐 layer 物理驻留量。Tombstone 或 digest
不匹配会被记为失败样本，harness 不会 fallback 到重新计算。

## SCBench

SCBench 位于 `benchmark/scbench/`。standard SCBench generation 使用 `run_scbench.py`，预处理 SCBench input 使用 `run_scbench_preprocessed.py`，KVZip 预处理对比使用 `run_kvzip_preprocessed.py`，生成 prediction 文件的评分使用 `compute_scores.py`。

```bash
python benchmark/scbench/run_scbench.py \
  --task scbench_qa_eng \
  --model_name_or_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --data_dir <SCBENCH_DATA_DIR> \
  --output_dir <OUTPUT_ROOT>/scbench \
  --attn_type dense \
  --kv_type dense \
  --ws 1 \
  --max_seq_length 131072 \
  --use_chat_template \
  --trust_remote_code
```

说明：

- SCBench 对 Sparse-Engine 方法使用原生 `sparseengine` attention type；通过
  `--hyper_param` 传入 method、checkpoint 和 keep budget。
- `compute_scores.py` 根据生成的 model tag 构造 prediction path。batch scoring 前应检查 runner output directory，不要假设手动选择的 path layout。
- `--num_eval_examples`、`--start_idx`、`--stop_idx`、shard 选项和 context-length filter 用于 partial/smoke run。记录结果时应将这些 output 标为 partial。
- `benchmark/scbench/run_kvzip_preprocessed.py` 是 KVZip 预处理 artifact 的路径；说明 preprocessing source 之前，不要用它替代 raw SCBench prediction。

## Claw-Eval

通过 Claw-Eval 评估 Sparse-Engine 模型时，使用 `benchmark/claw_eval/run_sparseengine_claw_eval.sh`。脚本准备或更新外部 `claw-eval` checkout，可选启动 `sparseengine.entrypoints.openai.api_server`，验证 Docker sandbox，渲染 Claw-Eval config，并写入 run manifest。

```bash
OPENROUTER_API_KEY=<KEY> \
MODEL_PATH=<MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
CUDA_VISIBLE_DEVICES=0 \
ENGINE_KWARGS='{"tensor_parallel_size":1,"gpu_memory_utilization":0.88,"max_model_len":131072,"engine_prefill_chunk_size":4096,"sparse_method":"vanilla"}' \
CLAW_EVAL_ARGS='batch --config ${CLAW_EVAL_CONFIG} --sandbox --trials 3 --parallel 1' \
bash benchmark/claw_eval/run_sparseengine_claw_eval.sh
```

使用已运行的 OpenAI-compatible server 时，禁用 managed server，并同时提供 API base URL 和 health endpoint：

```bash
START_SPARSEENGINE_SERVER=0 \
SPARSEENGINE_OPENAI_BASE_URL=http://127.0.0.1:18000/v1 \
SERVER_HEALTH_URL=http://127.0.0.1:18000/health \
SPARSEENGINE_OPENAI_API_KEY=local-sparseengine \
SPARSEENGINE_CLAW_MODEL_ID=sparseengine-claw \
CLAW_EVAL_ARGS='batch --config ${CLAW_EVAL_CONFIG} --sandbox --no-judge --trials 1 --parallel 1' \
bash benchmark/claw_eval/run_sparseengine_claw_eval.sh
```

说明：

- `--sandbox` 要求 Docker daemon 正在运行且 sandbox image 可用。runner 启动 preflight container，在评估前验证 `/health`、`/exec` 和 grader isolation。默认复用 `claw-eval-agent:latest`。image 不存在时，设置 `CLAW_EVAL_BUILD_SANDBOX_IMAGE=1` 显式构建；runner 不会静默启动大型 image build。
- 设置 `CLAW_EVAL_SANDBOX_IMAGE` 使用其他 image。`CLAW_EVAL_ARGS` 中显式的 `--sandbox-image` 优先。可通过 `CLAW_EVAL_DOCKER_BUILD_ARGS` 传入额外 `docker build` 参数。
- 设置 `CLAW_EVAL_UPDATE_REPO=0` 可以复用已存在且固定版本的外部 checkout，不 fetch 或更改它。解析后的 Claw-Eval commit 和 sandbox image ID/size 保存在 `run_manifest.json`。checkout 必须 clean；本地 source 改动会在评估前失败。
- 共享 OpenAI-compatible server 在该 benchmark path 中有意只支持文本。不支持的 OpenAI request field 会失败，而不是被静默忽略。设置 `CLAW_EVAL_TEXT_ONLY=1` 可生成记录在案的 task view，排除 `multimodal` task 以及暴露 image、PDF、presentation 或 spreadsheet 文件的 task。被排除项以 `skipped_by_policy` 行保留在 `per_sample_results.jsonl`；`final_summary.json` 分别记录 evaluated 和 skipped count。
- `CLAW_EVAL_ARGS` 必须包含且只包含一个显式 `--parallel`。对于 managed server，runner 将 `max_decoding_seqs` 设为 `parallel / SPARSEENGINE_DATA_PARALLEL_SIZE`，并拒绝不能整除的值。`max_num_seqs_in_batch` 仍是独立的 prefill batching control。
- 在同一 `RUN_NAME` 中继续中断的 run 时，把 `CLAW_EVAL_RESUME_TRACE_DIR` 设为该 run 的具体 trace subdirectory。runner 会记录该路径并传入 Claw-Eval 有界的 `--continue` 模式；拒绝当前 run 之外的 resume path。
- Remote launcher 可通过 `benchmark/ssh_reverse_tunnel.sh` 保持 reverse port forward。它使用 SSH keepalive，记录每次 reconnect，并在达到配置的 reconnect 上限后停止，不会隐藏永久中断的 route。
- 使用 `SETUP_ONLY=1` 可准备外部仓库、环境、Docker preflight、config、engine kwargs 文件和 run manifest，而不启动 model server 或 benchmark。setup-only 或 external-server 模式不要求 model path。
- 启用 judge 的 run 需要 `OPENROUTER_API_KEY`。开发 smoke run 可以传 `--no-judge`，但这些 score 不是官方 leaderboard result。官方 Claw-Eval 使用三次 trial 和 Pass^3。
- wrapper 验证当前 invocation 生成的 batch artifact。低 task score 是有效评估结果，但缺失 result、malformed trial 或 task-level error 会使 wrapper 非零退出。raw upstream 文件、规范化 `per_sample_results.jsonl` 和 `final_summary.json` 会保留在 run directory 供诊断。
- 默认输出位于 `<OUTPUT_ROOT>/claw-eval/<RUN_NAME>/`；需要时覆盖 `OUTPUT_ROOT` 或 `RUN_NAME`。

## 多模态

当前多模态 benchmark 入口和方法支持记录在 [`docs/zh/benchmarking/multimodal/`](multimodal/)。

## RULER core 与独立 NIAH

`benchmark/ruler_vt/pred.py` 是自包含 RULER core runner，支持
`niah_single_1`、`niah_multikey_2`、`vt`、`cwe` 和 `fwe`，通过 `--task`
选择。task contract 和 prompt 对齐
[NVIDIA RULER](https://github.com/NVIDIA/RULER)；为了不新增 `wonderwords` 和
大型 word asset 依赖，使用确定性的 synthetic word pool。因此这些 artifact 适用于
Sparse-Engine 回归比较，但不是官方 leaderboard dataset。依赖外部 essay 的 NIAH
以及 `qa_1`/`qa_2` 暂不包含。runner 生成 `dataset.jsonl`，保存 raw output、
parsed output、per-sample result、`run_info.json` 和 `aggregate_metrics.json`。

```bash
python benchmark/ruler_vt/pred.py \
  --task vt \
  --model-path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --output-dir <OUTPUT_ROOT>/ruler_vt/qwen25_7b_vanilla \
  --sparse-method vanilla \
  --context-lengths 4096,8192,16384 \
  --samples-per-length 20 \
  --hyper-param <ENGINE_PARAMS_JSON>
```

如需验证 prefix-cache correctness，请在 engine params 中设置
`enable_prefix_caching=true`，并增加 `--allow-prefix-caching
--prefix-cache-replay --require-prefix-cache-hit`。runner 会立即原样重放每个
deterministic batch，并要求 hit request/token 非零且输出完全一致。需要 matched
vanilla 对比和逐长度 quality grade 时，使用下文的 regression harness。

`benchmark/niah/test_niah.py` 是 needle-in-a-haystack utility runner。它可以在线生成 synthetic data，也可以使用 `--online_test=False` 加载 JSONL 文件。

```bash
python benchmark/niah/test_niah.py \
  --model_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --tokenizer_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --sparse_method vanilla \
  --output_path niah/qwen25_7b_vanilla \
  --context_lengths 16,32,64
```

## Regression Harness

固定 Sparse-Engine regression harness 记录在 [SparseVLLM 回归测试](sparseengine-regression-tests.md)中。
