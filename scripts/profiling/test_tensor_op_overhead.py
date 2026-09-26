import torch
import torch.nn as nn
import time
import fire


@torch.no_grad()
def test_overhead(batch_sizes=2048, use_compile=True, scenario="small", in_features=None, mid_features=None, out_features=None):
    """Measure tensor operation overhead with optional compilation and custom MLP dimensions."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")


    dtype = torch.bfloat16
    print(f"Dtype: {dtype}")


    if in_features is not None and mid_features is not None and out_features is not None:
        pass
    elif scenario == "llm":

        # Hidden=4096, Intermediate=14336
        in_features = 4096
        mid_features = 14336
        out_features = 4096
        print(f"Scenario: Llama-3-8B MLP simulation")
    else:

        in_features = 256
        mid_features = 4096
        out_features = 1024
        print(f"Scenario: Small Model (Default)")

    print(f"Dimensions: In={in_features}, Mid={mid_features}, Out={out_features}")


    if isinstance(batch_sizes, int):
        batch_sizes_list = [batch_sizes]
    elif isinstance(batch_sizes, str):

        if ',' in batch_sizes:
            batch_sizes_list = [int(x.strip()) for x in batch_sizes.split(',')]
        else:
            batch_sizes_list = [int(batch_sizes)]
    elif isinstance(batch_sizes, (list, tuple)):
        batch_sizes_list = batch_sizes
    else:
        batch_sizes_list = [batch_sizes]

    print(f"Batch sizes: {batch_sizes_list}")


    for batch_size in batch_sizes_list:
        print(f"\n{'='*20} Batch Size: {batch_size} {'='*20}")


        x = torch.randn(batch_size, in_features, device=device, dtype=dtype)


        model = nn.Sequential(
            nn.Linear(in_features, mid_features),
            nn.GELU(),
            nn.Linear(mid_features, out_features)
        ).to(device).to(dtype)

        def run_benchmark(target_model, label):
            print(f"\n--- Starting benchmark: {label} (BS={batch_size}) ---")


            warmup_iters = 50 if "Compile" in label else 20

            for _ in range(warmup_iters):
                _ = target_model(x)

            if device.type == 'cuda':
                torch.cuda.synchronize()


            iters = 1000


            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                for _ in range(iters):
                    _ = target_model(x)
                end_event.record()

                torch.cuda.synchronize()
                total_time_ms = start_event.elapsed_time(end_event)
                avg_time_ms = total_time_ms / iters
            else:

                start_time = time.perf_counter()
                for _ in range(iters):
                    _ = target_model(x)
                end_time = time.perf_counter()
                avg_time_ms = (end_time - start_time) / iters * 1000

            print(f"Mean iteration time: {avg_time_ms:.4f} ms")
            print(f"Iterations per second: {1000 / avg_time_ms:.2f} iterations/s")
            return avg_time_ms


        eager_time = run_benchmark(model, "Eager Mode")


        if use_compile:
            if hasattr(torch, 'compile'):
                print(f"\nCompiling model (torch.compile) - BS={batch_size}...")
                compiled_model = torch.compile(model)
                compile_time = run_benchmark(compiled_model, "Torch Compile")
                print(f"\n--- Comparison summary (BS={batch_size}) ---")
                print(f"Eager time: {eager_time:.4f} ms")
                print(f"Compile time: {compile_time:.4f} ms")
                print(f"Speedup: {eager_time / compile_time:.2f}x")
            else:
                print("\nCurrent PyTorch version does not support torch.compile")


if __name__ == "__main__":

    fire.Fire(test_overhead)
