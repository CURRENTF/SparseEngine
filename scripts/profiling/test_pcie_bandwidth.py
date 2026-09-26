import torch
import time

def test_bandwidth(size_mb=1024, device_id=0, warmup=True):
    """Measure CPU-to-GPU and GPU-to-CPU bandwidth for size_mb on device_id."""
    if not torch.cuda.is_available():
        print("Error: no CUDA device detected")
        return

    device = torch.device(f"cuda:{device_id}")


    n_bytes = size_mb * 1024 * 1024

    n_elements = n_bytes // 4

    try:


        cpu_tensor = torch.randn(n_elements, dtype=torch.float32).pin_memory()
        gpu_tensor = torch.empty(n_elements, dtype=torch.float32, device=device)


        if warmup:
            for _ in range(3):
                gpu_tensor.copy_(cpu_tensor, non_blocking=True)
                cpu_tensor.copy_(gpu_tensor, non_blocking=True)
            torch.cuda.synchronize()


        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        gpu_tensor.copy_(cpu_tensor, non_blocking=True)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        h2d_bandwidth = (n_bytes / (1024**3)) / (elapsed_time_ms / 1000)


        start_event.record()
        cpu_tensor.copy_(gpu_tensor, non_blocking=True)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        d2h_bandwidth = (n_bytes / (1024**3)) / (elapsed_time_ms / 1000)

        print(f"| {size_mb:>7} MB | {h2d_bandwidth:>12.2f} | {d2h_bandwidth:>12.2f} |")


        del gpu_tensor
        del cpu_tensor
        torch.cuda.empty_cache()

    except RuntimeError as e:
        print(f"| {size_mb:>7} MB | {'OOM/Error':>12} | {'OOM/Error':>12} |")
        # print(f"Error detail: {e}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"Testing device: {device_name}")
        print(f"{'-'*46}")
        print(f"|    Size     | H2D (GB/s)   | D2H (GB/s)   |")
        print(f"|{'-'*13}|{'-'*14}|{'-'*14}|")

        sizes = [32, 64, 128, 256, 512, 1024, 2048]
        for size in sizes:
            test_bandwidth(size_mb=size, device_id=0, warmup=True)
        print(f"{'-'*46}")
    else:
        print("No CUDA device detected; cannot run benchmark.")
