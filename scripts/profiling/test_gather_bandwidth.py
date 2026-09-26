import torch
import time
import argparse

torch.set_num_threads(8)

def test_gather_and_transfer(
    dim0=20,
    dim1=2000000,
    dim2=256,
    dtype_str="float16",
    test_sizes=[128, 1024, 4096, 8192, 20000],
    num_iters=10
):
    if not torch.cuda.is_available():
        print("Error: CUDA not available.")
        return


    dtype = getattr(torch, dtype_str)
    element_size = torch.tensor([], dtype=dtype).element_size()
    total_elements = dim0 * dim1 * dim2
    total_mem_gb = (total_elements * element_size) / (1024**3)

    print(f"Initializing tensor [{dim0}, {dim1}, {dim2}] | dtype: {dtype_str} | Iterations: {num_iters}")
    print(f"Estimated memory usage: {total_mem_gb:.2f} GB")

    try:

        src_tensor = torch.empty(dim0, dim1, dim2, dtype=dtype).pin_memory()
    except RuntimeError as e:
        print(f"Insufficient memory to create tensor: {e}")
        return

    device = torch.device("cuda:0")
    print("Tensor initialized. Starting benchmark...")
    print(f"{'-'*110}")
    print(f"| {'Test Size':<10} | {'AdvIndex(Avg) ms':<18} | {'IdxSel(Avg) ms':<16} | {'E2E(Avg) ms':<18} | {'Eff. BW (GB/s)':<15} |")
    print(f"|{'-'*12}|{'-'*20}|{'-'*18}|{'-'*20}|{'-'*17}|")

    for size in test_sizes:

        indices = torch.randint(0, dim1, (size,), dtype=torch.long)


        for _ in range(3):
            _ = src_tensor[:, indices, :]
            _ = torch.index_select(src_tensor, 1, indices)


        adv_times = []
        for _ in range(num_iters):
            start_time = time.time()
            _ = src_tensor[:, indices, :]
            adv_times.append((time.time() - start_time) * 1000)
        avg_adv_ms = sum(adv_times) / len(adv_times)


        is_times = []
        for _ in range(num_iters):
            start_time = time.time()
            _ = torch.index_select(src_tensor, 1, indices)
            is_times.append((time.time() - start_time) * 1000)
        avg_is_ms = sum(is_times) / len(is_times)


        e2e_times = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            gathered_temp = torch.index_select(src_tensor, 1, indices)
            gpu_tensor = gathered_temp.to(device, non_blocking=True)
            end_event.record()

            torch.cuda.synchronize()
            e2e_times.append(start_event.elapsed_time(end_event))

        avg_e2e_ms = sum(e2e_times) / len(e2e_times)


        transferred_elements = dim0 * size * dim2
        transferred_bytes = transferred_elements * element_size
        eff_bandwidth = (transferred_bytes / (1024**3)) / (avg_e2e_ms / 1000) if avg_e2e_ms > 0 else 0

        print(f"| {size:<10} | {avg_adv_ms:>18.3f} | {avg_is_ms:>16.3f} | {avg_e2e_ms:>18.3f} | {eff_bandwidth:>15.2f} |")

    print(f"{'-'*110}")

    del src_tensor
    del indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CPU Gather and H2D Bandwidth")
    parser.add_argument("--dim0", type=int, default=20, help="Batch size (dim 0)")
    parser.add_argument("--dim1", type=int, default=2000000, help="Sequence length (dim 1)")
    parser.add_argument("--dim2", type=int, default=256, help="Hidden size (dim 2)")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16", "int8"], help="Data type")
    parser.add_argument("--iters", type=int, default=10, help="Number of iterations for averaging")

    args = parser.parse_args()


    test_sizes = [100, 1000, 5000, 10000, 20000]

    test_gather_and_transfer(args.dim0, args.dim1, args.dim2, args.dtype, test_sizes, args.iters)
