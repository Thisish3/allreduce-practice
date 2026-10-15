"""Is a "non_blocking" H2D copy actually asynchronous here?

For each host-memory kind and stream, time:
  call_ms   CPU time until copy_() returns (async DMA -> close to 0)
  event_ms  GPU-side duration between CUDA events around the copy
  wall_ms   call + wait until the copy is done
"""

import json
import statistics
import time

import torch

N = 32768 * 4096 * 2  # 256 MiB
REPS = 7


def measure(host, dev, stream):
    calls, events, walls = [], [], []
    for i in range(REPS + 1):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            t0 = time.perf_counter()
            start.record()
            dev.copy_(host, non_blocking=True)
            end.record()
            t1 = time.perf_counter()
        end.synchronize()
        t2 = time.perf_counter()
        if i:  # skip warm-up
            calls.append((t1 - t0) * 1e3)
            events.append(start.elapsed_time(end))
            walls.append((t2 - t0) * 1e3)
    med = statistics.median
    return {"call_ms": round(med(calls), 2), "event_ms": round(med(events), 2),
            "wall_ms": round(med(walls), 2), "event_gbps": round(N / (med(events) * 1e-3) / 1e9, 2)}


def main():
    dev = torch.empty(N, dtype=torch.uint8, device="cuda")
    hosts = {
        "pin_memory_alloc": torch.empty(N, dtype=torch.uint8, pin_memory=True),
        "dot_pin_memory": torch.randint(0, 255, (N,), dtype=torch.uint8).pin_memory(),
        "pageable": torch.randint(0, 255, (N,), dtype=torch.uint8),
    }
    streams = {"default": torch.cuda.default_stream(), "side": torch.cuda.Stream()}
    # same bytes as overlap.py's tensors: fp16 [32768, 4096]
    dev16 = torch.empty(32768, 4096, dtype=torch.float16, device="cuda")
    hosts["fp16_2d_dot_pin_memory"] = torch.randn(32768, 4096, dtype=torch.float16).pin_memory()
    out = {"gpu": torch.cuda.get_device_name(), "bytes": N, "cases": {}}
    for hn, h in hosts.items():
        for sn, s in streams.items():
            r = measure(h, dev16 if h.dtype == torch.float16 else dev, s)
            out["cases"][f"{hn}/{sn}"] = r
            print(f"{hn:18s} {sn:8s} {json.dumps(r)}", flush=True)
    with open("results/diagnose.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
