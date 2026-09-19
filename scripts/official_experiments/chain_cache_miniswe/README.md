# GLM-4.7-Flash MiniSWE: cross-request cache reuse

This recipe runs real mini-SWE-agent closed-loop tasks through
`benchmark/swe_bench_lite/run.py`, followed by the official SWE-bench Lite
Docker evaluator. It does not run synthetic or teacher-forced generation.
Use `setting.json` as the single experimental setting. No GPU result is implied
by this recipe; every method needs its own smoke and concurrency pilot.

## Frozen comparison

- Same BF16 GLM-4.7-Flash checkpoint, TP2/EP2/DP1, two GPUs, target 64 agent workers.
  Agent concurrency and server prefill/decode/resident limits are separate; record
  both, and select each through pilot evidence. Check the checkpoint dtype in the
  saved model config; do not substitute an FP8 checkpoint within the campaign.
- Context limit 202752; output limit 16384; temperature .7, top_p 1; 80 agent
  steps; 7200 seconds per task, including model and tool waits. These sampling
  values follow the [GLM-4.7-Flash model card](https://huggingface.co/zai-org/GLM-4.7-Flash#evaluation-parameters)
  SWE-Bench Verified and Terminal Bench recommendation. All six methods read
  this one global agent setting and may not override it. The shared API adapter
  has no seed control. Keep the
  upstream `swebench.yaml` prompt identical and save its version through the
  canonical manifest. Timeouts remain failures in the score denominator.
- Thinking is enabled for every method. [Preserved Thinking](https://docs.z.ai/guides/capabilities/thinking-mode#preserved-thinking)
  is also enabled for
  every method by returning unmodified `reasoning_content` and rendering prior
  assistant reasoning with `clear_thinking=false`. Do not rely on the model
  template's default or change thinking behavior for an individual method.
- On `finish_reason=length` followed by MiniSWE `FormatError`, the shared adapter
  retains the truncated assistant turn before the upstream correction for every
  method. Renderable tool calls receive explicit "not executed" tool results;
  truncated arguments are never executed by this recovery path. The original
  response stays on the correction only, with `truncated_response_preserved=true`,
  so usage is recorded once. This changes subsequent closed-loop prompts and
  requires a fresh run for comparisons with the previous recovery behavior.
  Chain continuation after `length` uses full token-prefix validation, without
  the append shortcut: incomplete tool serialization or tokenization differences
  may still require the server to release and recreate the chain safely.
- Prefill chunk 4096, batch token budget 65536, GPU memory utilization .95, and
  CUDA Graph on. CPU cache offload is disabled for the frozen comparison.
  Actual concurrent decoding varies while agents run tools. The default is
  **64 agent workers**, not proof of 64 resident requests or the maximum feasible
  server limit.
- Full selection is the canonical sorted Lite test set of 300. Pilot uses its
  first max(64, concurrency); smoke uses the first 1. Compare `instances.txt` across methods.
  `batch_size=300` avoids the old 50-task batches capping 64 workers at 50.

| Label | Cache | Budget and exceptions |
|---|---|---|
| snapkv-chain | chain | 64 sink + 512 recent + 15808 selected = 16384 |
| h2o-chain | chain | prefill 16384, decode 8192; online eviction every 128 steps |
| omnikv-prefix | radix | 64 + 512 + 1472 = 2048; model-profile full layers retained |
| quest-prefix | radix | total selection 2048; page size 16; first 2 layers full |
| vanilla-prefix | radix | dense KV |
| snapkv-no-chain | disabled | exactly the same SnapKV settings, cache reuse disabled |

SnapKV uses probability scoring/window 32. H2O uses probability scoring/window
128, recent ratio .5, FP32 scores and online cumulative updates. Current MLA H2O
reduces heads before softmax in decode: it is not official per-head H2O parity.
Do not silently switch to logits or disable online eviction to get a passing run.
OmniKV/Quest selection budgets are not physical KV storage limits, and full layers
are exceptions. These are useful operating points, not equal-quality algorithms.

## Preparation and execution

### Upstream vLLM baseline

`vanilla-prefix` normally means **Sparse-Engine dense attention**, not upstream
vLLM. For an upstream comparison, prepare a separate root with
`prepare --backend vllm --concurrency C --engine-concurrency S` and a setting
copy containing the requested TP/DP/EP topology. Then use the same `serve`,
`bench`, and `collect` commands with an activated vLLM environment for `serve`.
Only `vanilla-prefix` is supported by this backend. The manifest identifies
the backend, installed version, exact command, and effective engine settings.

For TP1/DP2/EP2 with 16 client agents, `--engine-concurrency 8` maps to
vLLM `max_num_seqs=8` per replica. vLLM uses its native prefix cache, chunked
prefill, and CUDA Graph scheduling. Sparse-Engine resident rows, decode reservation,
fixed prefill chunk size, and explicit decode Graph buckets are not vLLM knobs
and are omitted from its effective config. Batch token budget, context limit,
memory fraction, model dtype, and all client sampling settings remain explicit.
These engine differences must be retained when interpreting comparisons.

The vLLM launcher enables prompt token details and an ASGI middleware that saves
unaltered non-streaming requests/responses for `collect`. Request time includes
server queueing and response delivery; it is not isolated GPU execution time.
Probe `/v1/models` for readiness and `/metrics` for vLLM queue, cache, and
preemption diagnostics. Sparse-Engine `/v1/worker/load` does not exist on vLLM.

### Shared workflow

All paths below are supplied by the operator. Use a persistent data disk for
`RUN_ROOT`, check free space (including Docker storage), and record the choice.
Use another persistent data disk if the preferred disk has insufficient space. Do not download
images/models implicitly. The scripts refuse to overwrite invocation artifacts.
For a repaired attempt, prepare a new root and preserve the previous failure.

```bash
RECIPE=scripts/official_experiments/chain_cache_miniswe/run.py
python3 "$RECIPE" prepare --root "$RUN_ROOT" --concurrency "$CONCURRENCY"
```

`--concurrency` controls the number of concurrent MiniSWE agents. By default the
server prefill, decode, and resident-sequence limits match it. When the client
concurrency intentionally exceeds the number of sequences that must reside on
the GPUs at once, set the server limit explicitly, for example
`--concurrency 64 --engine-concurrency 48`. Record both values in the result;
they are different workload and engine controls.

For the 24-agent candidate with 64 resident rows and offload disabled, prepare
a fresh root with the shared `.95` memory setting:

```bash
python3 "$RECIPE" prepare --root "$RUN_ROOT" --concurrency 24 \
  --engine-concurrency 24 --engine-resident-concurrency 64
```

This keeps prefill/decode sequence limits at 24 and captures decode Graphs only
through batch 24. The 64 resident rows allow additional idle chains to retain
their GPU state. `gpu_memory_utilization=.95` is the memory planning fraction,
not a target for measured GPU compute utilization. This configuration still
needs a new pilot; earlier `.90` runs do not establish its stability.

The shared engine setting uses `decode_reservation_tokens=1024`, a rolling
reservation window for future decode capacity. It does not change the agent's
16384-token output limit. For the 32-agent, 64-resident-row candidate, use
`--concurrency 32 --engine-concurrency 32 --engine-resident-concurrency 64` in
a fresh root; retain `.95` memory utilization and disabled offload.

If 64 agents must be admitted while only 48 sequences decode in one engine
step, keep 64 resident rows without increasing the decode Graph batch:
`--concurrency 64 --engine-concurrency 48 --engine-resident-concurrency 64`.
This lets the scheduler queue work behind the 48-sequence execution limit.
Setting resident rows to 48 instead causes excess simultaneous chain admissions
to return HTTP 503 `chain_capacity_unavailable`; client retries are bounded and
do not make that configuration a stable 64-agent result.

Start with `snapkv-chain`, then `h2o-chain`, `omnikv-prefix`, `quest-prefix`,
`vanilla-prefix`, `snapkv-no-chain`. Finish smoke for each before its pilot;
complete pilots before choosing the formal queue. Run methods sequentially
on the same GPU pair. Each agent/server concurrency pair gets a fresh root, e.g.
`agents64-engine48`.
The prepared settings freeze both agent and engine limits; never edit an active root.
Changing the output limit from 8192 to 16384 invalidates earlier concurrency
pilots for capacity selection. Prepare a new root and rerun smoke/pilot before
using an old stable-concurrency claim; in particular, the prior c24/8192 pilot
does not establish c24 capacity under this protocol.

Concurrency plan: for radix methods start at 4 or 8 and double toward 64;
for SnapKV/H2O start at 16, then 32/48/64, and only if stable try 96/128.
A one-task smoke tests interfaces only. A complete pilot at the intended worker
count tests long-lived sessions and includes at least that many tasks. Inspect
actual running/decoding counts, cache reuse, preemption, idle-chain eviction and
HTTP failures, not just process exit. Short successful tasks do not establish a
worst-case cache capacity. Stop escalation at memory/admission failure; ordinary
bugs require diagnosis and must not be called capacity limits. Keep full logical
history and all length/budget settings unchanged during this scan.

Choose a common pilot-validated concurrency for a matched-concurrency comparison,
and optionally each method's largest pilot-validated concurrency for application
capacity results. Label the latter "largest tested", not an integer maximum
unless the next integer was actually tested under the same workload. SnapKV
on/off pilots for the slow-cost gate must be at the same concurrency and share
one prepared root. Differences in full-run concurrency must remain visible in
the results and cannot be attributed solely to cache reuse.

GPU host, in a persistent tmux session, after activating the server conda env
(`conda activate "$SERVER_ENV"`; do not invoke only a conda Python without
activation). `SERVER_PYTHON` should be `command -v python` in that environment:

```bash
python3 "$RECIPE" serve --root "$RUN_ROOT" --method "$METHOD" --phase "$PHASE" \
  --python "$SERVER_PYTHON" --model "$GLM47_MODEL" --gpus "$GPU_PAIR" \
  --port 18147 --timeout 86400
```

The benchmark coordinator must install an exit trap before starting the stages,
so success, failure, timeout, and operator interruption all stop the task-owned
server. The stop command validates hostname, boot ID, PID start time, and private
process group before sending a signal; it never searches or kills by process name:

```bash
cleanup_server() {
  python3 "$RECIPE" stop --root "$RUN_ROOT" --method "$METHOD" --phase "$PHASE"
}
trap cleanup_server EXIT INT TERM
```

For a remote Docker worker, keep the trap in the GPU-host coordinator, but wrap
`wait-remote` below, **not a foreground SSH benchmark command**. SSH exit 255 is
a transport failure, not evidence that the remote task finished. A worker-side
trap cannot stop the GPU-host server. Stop after confirmed generation success
with `stop --reason generation_completed`; use `--reason coordinator_failure`
in the failure trap. `server.result.json` records the supplied reason; a successful
requested stop does not prove benchmark success. Confirm both selected GPUs have
no task-owned compute processes before advancing to the next phase.

When launching from a service-hosted terminal or coding agent, put the existing
coordinator script in an independent user systemd service. A newly started tmux
server or `setsid` process can still inherit the launching service's cgroup;
restarting that service can terminate the entire experiment.

```bash
systemd-run --user --unit "$RUN_UNIT" --property=Type=exec \
  --property=KillMode=mixed --property=TimeoutStopSec=1800 \
  --property=Restart=no --property=RuntimeMaxSec=100000 \
  --working-directory "$REPO_ROOT" --setenv="PATH=$PATH" \
  /bin/bash "$COORDINATOR_SCRIPT"
systemctl --user show "$RUN_UNIT" -p MainPID -p ControlGroup -p ActiveState
```

The coordinator still activates its conda environment and invokes this recipe;
this changes process ownership only. Use a unique unit and fresh artifact root
for each attempt. Check `/proc/<pid>/cgroup` for the coordinator, both model ranks,
and tunnel: they must belong to the experiment unit, outside the calling service.
`KillMode=mixed` gives the coordinator's TERM trap time to clean up its children;
size the stop timeout to cover bounded cleanup and artifact-transfer retries.
Keep run/status/result files on the data disk, since the unit may be unloaded
after exit. Remote benchmark stages retain their worker-owned tmux sessions.

For unattended user services, also verify `loginctl show-user "$USER" -p Linger`.
Enable lingering for the experiment account with `loginctl enable-linger "$USER"`
when permitted, then verify `Linger=yes` before launch. An independent unit alone
does not keep its user service manager alive after the final login session ends.
Lingering does not protect against explicitly stopping the user manager, killing
the experiment, or rebooting the machine; preserve interrupted runs as failures.

`GPU_PAIR` must contain two distinct idle indices. The launcher checks compute
PIDs, used memory and utilization, and refuses an occupied port. Recheck ownership
during startup until both ranks attach; the preflight check is not a GPU
reservation. Never kill another process or pass occupied GPUs as "idle".
Record PID/UUID/start time. The foreground launcher owns only its child process
group; interrupting it stops that server group. Stop it after each phase and
start a fresh server for the next phase so pilot caches cannot warm full tasks.

If the artifact root is network-mounted, pass `serve --compile-cache-root
"$LOCAL_COMPILE_ROOT"` to place Triton/CUDA/FlashInfer caches and compiler
temporary files on a verified node-local filesystem. Check the mount type,
available space and write permission first; the runner does not choose a disk
automatically. A task-owned tmpfs directory is also possible when RAM capacity
is sufficient, but its contents are not durable across reboot. Keep results on
the data disk. Effective compiler paths are saved in `server_manifest.json`.
This removes network filesystem I/O from compilation; it does not by itself
prevent new kernel specializations. Use a fresh run identity to deploy source
or launch changes rather than modifying a running evaluation.

Docker driver, in its activated environment, after `/readyz` returns success:

```bash
python3 "$RECIPE" bench --root "$RUN_ROOT" --method "$METHOD" --phase "$PHASE" \
  --python "$SWE_PYTHON" --swe-bench-dir "$SWE_BENCH_DIR" \
  --api-base http://127.0.0.1:18147/v1 --stage prepare --timeout 1800
python3 "$RECIPE" bench --root "$RUN_ROOT" --method "$METHOD" --phase "$PHASE" \
  --python "$SWE_PYTHON" --swe-bench-dir "$SWE_BENCH_DIR" \
  --api-base http://127.0.0.1:18147/v1 --stage generate --timeout 43200
python3 "$RECIPE" bench --root "$RUN_ROOT" --method "$METHOD" --phase "$PHASE" \
  --python "$SWE_PYTHON" --swe-bench-dir "$SWE_BENCH_DIR" \
  --api-base http://127.0.0.1:18147/v1 --stage evaluate --timeout 43200
python3 "$RECIPE" bench --root "$RUN_ROOT" --method "$METHOD" --phase "$PHASE" \
  --python "$SWE_PYTHON" --swe-bench-dir "$SWE_BENCH_DIR" \
  --api-base http://127.0.0.1:18147/v1 --stage summarize --timeout 1800
```

Use `PHASE=smoke`, then `pilot`, then `full`; pilot/full use the prepared concurrency.
The full campaign has one official 300-task evaluation per method. A paper claim
about stable performance needs repeated independent campaigns, not duplicated
aggregates. Preflight Docker images and host RAM/process/storage capacity for 64
containers before generation. `prepare` uses the canonical offline dataset/image
checks. Evaluation uses 16 workers and can run after stopping the GPU server.

If local Docker is unavailable, use the established Docker worker. Copy this
recipe, the same adapter revision, the prepared root's settings/engine JSONs,
and each phase's `server_manifest.json` to that host. Keep the directory structure;
paths to Python/SWE-bench and `--root` can differ. Establish an SSH reverse tunnel
to the local server and use the forwarded URL as `--api-base` consistently within
each run. Do not start a second GPU server remotely. Copy benchmark artifacts
back beside the original server logs before `collect`. Server readiness alone is
insufficient: verify both rank logs, actual Graph execution, and at least two
successful turns with nonzero reuse for the enabled cache modes.

### Remote connection recovery

Use an installed `autossh` for the task-owned tunnel. Keepalive settings detect
disconnects; plain `ssh -N -R` does not reconnect. Keep the autossh PID in the
coordinator's exit cleanup, and do not reuse or stop another task's tunnel:

```bash
AUTOSSH_GATETIME=0 autossh -M 0 -N \
  -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -R "127.0.0.1:$REMOTE_PORT:127.0.0.1:$SERVER_PORT" \
  "$WORKER" > "$RUN_ROOT/tunnel.log" 2>&1 &
tunnel_pid=$!
```

Add the site's SSH port/jump-host options when required. With autossh, use
`-o ProxyJump=...` rather than `-J ...`: older autossh argument parsers may reject
the latter even when the installed ssh supports it. On the worker, use the
same activated-environment launcher for `bench`, accepting a stage and forwarding
the remaining arguments (including `--detach`):

```bash
stage=${1:?stage required}
shift
# Activate the worker conda environment before this existing entrypoint.
python "$RECIPE" bench --root "$REMOTE_ROOT" --method "$METHOD" --phase "$PHASE" \
  --python "$(command -v python)" --swe-bench-dir "$SWE_BENCH_DIR" \
  --api-base "http://127.0.0.1:$REMOTE_PORT/v1" \
  --stage "$stage" --timeout "$STAGE_TIMEOUT" "$@"
```

`bench --detach` starts the stage once in a worker-owned tmux session; subsequent
calls query that same stage. It writes `*.detached.json`, `*.detached.log` and an
atomic `*.detached.result.json`, including preflight failures. A lost launch reply
cannot start a second benchmark. Existing failed stages are never automatically
rerun. Keep the stage-specific pilot timeout from the slow-baseline policy.

The GPU-host coordinator waits through transient control/tunnel failures:

```bash
python "$RECIPE" wait-remote --root "$RUN_ROOT" --method "$METHOD" --phase "$PHASE" \
  --stage "$STAGE" --ssh-command "$SSH_COMMAND" \
  --worker-command "bash $REMOTE_LAUNCHER $STAGE --detach" \
  --reconnect-attempts 30 --poll-interval 10 --recovery-timeout 1800 --timeout 45000
```

`SSH_COMMAND` is shell-quoted SSH argv including the destination and required
options; `worker-command` is likewise shell-quoted argv. These are parsed as
arguments, not local shell programs. Use `wait-remote` for each stage. It logs all
polls to `*.remote.events.jsonl` and retries SSH exit 255/probe timeout. Worker-side
API readiness gates only the initial prepare/generate launch; a failed startup
probe records its exception and waits within the recovery budget. Once launched,
polls check the worker session and terminal result without probing the API.
HTTP readiness failure does not establish task failure: inference may still be
progressing. A `running` reply resets the connection budget even if an older
worker includes `api_ready=false` in that reply.

A confirmed task failure fails immediately; connectivity loss preserves the
local server while this command waits. Each control-connection outage permits
30 further attempts, spaced 10 seconds apart. It exits nonzero after actual
control retries are exhausted, startup readiness/recovery expires, or the total
stage deadline expires, allowing the coordinator to clean up its server. A running
worker remains subject to that total deadline. Record the failure and inspect
remaining worker tasks/containers; a deadline with unknown remote state is not
completion. Never add API-health-triggered cleanup while the worker is running.

After generate succeeds, stop the server, then run evaluate/summarize through the
same detached path (these stages do not require a live API). Copy results back
before collect; an interrupted artifact transfer may be repeated without rerunning
generation. Reconnection cannot rescue already disconnected HTTP requests, so
continue counting cancellations, HTTP 410 and full-history recovery work.

## Throughput diagnostics

For a diagnostic run, set `engine.throughput_log_interval_s` to a positive
interval in the setting file before `prepare`; the API server otherwise disables
these periodic logs. The log reports computed prefill tokens, decode tokens,
prefill step count and `decode_batch_steps` (executed batch size to step count),
alongside queue lengths. A decode queue of 40 does not prove execution at batch 40.
Both token rates use the entire wall-clock interval, including other stages and
idle periods; they are not isolated prefill/decode execution throughput.

`engine.enable_profiler=true` adds aggregate host-observed section timings.
Keep `SPARSEENGINE_SYNC_DEVICE` and `CUDA_SYNC_SENGINE` disabled for ordinary serving
measurements. Nested profiler sections overlap and asynchronous CUDA work is not
fully attributed by host timings. Correlate these diagnostics with timestamped
GPU activity/power, worker load, request logs and task timing; use a separate GPU
timeline capture when kernel or communication attribution is required. Record
diagnostic settings and restart the task-owned server between phases.

For a separately labelled DP-attention diagnostic, supply a setting file with
`tensor_parallel_size=1`, `data_parallel_size=2`, `expert_parallel_size=2` and
`moe_backend=agrs`. Engine sequence limits apply per DP replica: for 16 agent
workers and an execution limit of 8 per replica, prepare with `--concurrency 16
--engine-concurrency 8`. Resident-row capacity is also per replica. Keep sampling
settings unchanged and record this topology separately from the frozen TP2/EP2
comparison. Changing topology or concurrency requires a new prepared root.

## Reverse SSH health

An established SSH connection or a listening relay port does not establish
end-to-end health. Probe the forwarded SSH banner and verify an authenticated
worker command before launching the model server. If rebuilding the connection
does not restore the banner, inspect the worker's SSH service and resource state;
do not assume that further tunnel resets will repair it.

`reverse_ssh_watchdog.py` is an optional relay-side probe for explicitly dedicated
loopback reverse-SSH ports. It waits for three consecutive failed probes before
terminating the listener's SSH session, with at most 30 rebuilds per continuous
outage. The worker must already have a reconnect service. The helper records
failures and rebuilds in `--state`; it does not restart benchmark samples.
It refuses listeners with an unexpected owner, executable, or additional listening
sockets and pins process identity before signalling. These checks cannot detect
other experiments using channels through the same port: operators must establish
exclusive use before enabling automatic resets. Keep only one recovery controller
per connection. A healthy banner still requires authenticated/API verification.

## Slow baseline policy and failures

Run SnapKV-no-chain last. For its **64-task pilot generation only**, use
`--timeout 7200`. Generate the SnapKV-chain pilot first with the same agent
settings. The `full --stage generate` command automatically skips the no-chain
baseline if its pilot timed out, its elapsed-time ratio exceeds 4, or its rough
300-task projection exceeds 12 hours. Pilot selection scales to at least the
prepared concurrency. The decision is saved as
`slow_baseline_decision.json` with `skipped_by_policy`; this is a cost policy,
not a performance conclusion. Use separate `generate` and `evaluate` commands
for pilots, because the gate reads generation time, excluding evaluation.
If the pilot times out, preserve its timeout record, stop its server, and check
for task-owned leftover Docker containers before proceeding. Do not evaluate an
incomplete pilot as a complete score. To record the full skip, invoke the full
generation command; it checks the gate before contacting a server.

Ordinary crashes, HTTP errors, prefix mismatches, OOM, timeouts on other methods,
and invalid outputs are explicit failures. Do not call them capacity boundaries,
slow-baseline skips, or zero-quality results. Preserve logs and move to other
methods; report failures in the final campaign record. An idle-chain HTTP 410
can use only the adapter's existing bounded recovery, which must be counted as
recomputation. No new retry/fallback is enabled here.

The Docker writable-layer monitor treats a container that disappears before
agent cleanup as a failure of that sample. Valid size records for other
containers are still checked; an exact `no such object` response must not fail
the whole batch. Unattributed monitoring failures remain subject to bounded
fail-closed handling. Failed container cleanup is recorded separately and must
not terminate monitoring for other samples.

## Evidence and interpretation

```bash
python3 "$RECIPE" collect --root "$RUN_ROOT" --method "$METHOD" --phase "$PHASE"
```

Collection validates selected IDs and exact smoke/pilot/full row coverage, preserves the
official summary, and exports `request_samples.jsonl` and `report.json` from
server request JSON. Also retain server logs, trajectories, predictions, official
reports, and environment manifests. `prepare` records the Git commit and working-tree
status in `source.json`; it does not copy source files or save a working-tree patch.
Use the Research-Vault workflow for the final dated record and compact data.

`timed_mini.py` delegates to the installed `mini-extra` console entrypoint and
wraps only its `process_instance` call. It preserves returned values/exceptions
and writes one `*.wall_time.json` per task. Unsupported upstream module/signature
changes fail explicitly. This uses the upstream
[SWE-bench instance boundary](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/run/benchmarks/swebench.py);
verify the installed version during smoke. Task time includes environment startup,
model waiting and tool execution, excludes executor-queue waiting and official
evaluation. An interrupted task may have no final timing: collection rejects
missing coverage. Keep timeout/failed tasks distinct from successful completions.
`task_samples.jsonl` joins these times to official outcomes; `report.json` reports
all-attempt and resolved-only P50/P95. Resolved-only subsets differ across methods
and are descriptive, not paired comparisons. Generate/evaluate wall times come
from separate `*.result.json` records; do not use a combined `all` stage.

Report resolved/300, all failure statuses, API calls, reused tokens, logical
uncached prompt tokens, generation wall time and summed server request duration.
Summed request durations overlap at concurrency 64: they are neither GPU time
nor workload wall time. Logical uncached tokens do not include unobserved
scheduler recompute and cannot be called exact executed-prefill tokens.

The existing nonstreaming API logs **do not measure TTFT/TPOT**. Collection
explicitly emits null/status instead of deriving them from full-response time.
It also does not claim peak KV usage or H2O state equality. For those additional
experiment-3 figures, add validated engine-event observation/fixed-trace replay
and an independent state-restoration oracle in a follow-up. This campaign tests
closed-loop application utility: same issues, but method-dependent generated
tokens, tool results and turn counts. Do not plot its latency ratios as a
same-input Chain Cache speedup. Inspect resets/reuse and separate generation
failures from official test failures before attributing score differences.
