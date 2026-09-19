# Nsight Compute Playbook

Use Nsight Compute after correctness and stable microbenchmarks narrow the
candidate set. Nsight replay and metric collection can perturb execution, so
do not use profiler duration as the production latency result.

## Capture Deliberately

1. Choose one representative shape and a callable that launches the target
   kernel predictably.
2. Warm JIT compilation before capture.
3. Filter the target kernel or limit launches when possible.
4. Start with focused sections, then collect a broader set only when needed.
5. Save the command, console output, and `.ncu-rep` artifact.
6. Capture baseline and candidate under the same software and hardware state.

Useful section families include Speed of Light, Launch Statistics, Occupancy,
Memory Workload Analysis, Scheduler Statistics, Warp State Statistics, and
Source Counters. Confirm the exact section names supported by the installed
`ncu` version before scripting them.

## Interpret as a Chain of Evidence

### Launch and Occupancy

Check grid size, waves per SM, block/thread shape, shared memory, registers per
thread, theoretical occupancy, and achieved occupancy. Low occupancy matters
only when it limits latency hiding or parallelism; high occupancy does not
prove efficiency.

### Memory

Compare achieved DRAM/L2/shared throughput, transaction efficiency, cache hit
rates, sectors, and shared-bank conflicts. Relate bytes moved to the algorithm
and wrapper, including intermediate tensors eliminated or introduced by
fusion.

### Compute

Check tensor-core or arithmetic-pipe utilization, instruction mix, issue rate,
and dependency stalls. Verify that the chosen tile and dtype actually reach
the intended hardware path.

### Registers and Stalls

Inspect register count, local-memory traffic, spills, scoreboard/dependency
stalls, barrier stalls, memory throttling, and not-selected warps. Connect a
stall change to a concrete code or schedule change before acting on it.

### Roofline

Estimate arithmetic intensity using the measured callable boundary. Classify
memory- versus compute-limited behavior only when achieved bandwidth/compute
and the traffic model agree. Re-evaluate after fusion because the boundary and
bytes moved have changed.

## Close the Loop

Use the profile to form one next hypothesis, benchmark the resulting change,
and retain it only if stable latency improves without breaking correctness.
Do not optimize a metric that does not move the microbenchmark or end-to-end
result.
