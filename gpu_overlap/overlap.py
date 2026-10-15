"""Hide host->device transfers behind GEMMs with CUDA streams (one GPU).

The GPU analog of "overlap send with reduce": data has to cross a link (PCIe)
before it can be computed on. Measures, with torch on one GPU:

  1. link bandwidth   H2D / D2H, pinned vs. pageable host memory
  2. two streams      copy and compute that do not depend on each other:
                      alone, back-to-back on one stream, concurrently on two
  3. chunk pipeline   compute depends on its copy; split into C chunks and
                      copy chunk i+1 while computing chunk i

Durations of single operations are GPU-side (CUDA events); end-to-end runs
are wall time with the GPU synced. Each measurement drops WARMUP runs and
reports the median of REPS.

    python overlap.py [run_id]     # writes results/overlap_<gpu>_<run_id>.json
"""

import json
import os
import statistics
import sys
import time

import torch

HIDDEN = 4096
ROWS = 32768            # 32768 x 4096 fp16 = 256 MiB to move
WARMUP = 3
REPS = 7
CHUNKS = [1, 2, 4, 8, 16, 32]


def wall(fn):
    """Median wall time (ms) of fn(), GPU synced around each run."""
    runs = []
    for i in range(WARMUP + REPS):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        if i >= WARMUP:
            runs.append((time.perf_counter() - t) * 1e3)
    return round(statistics.median(runs), 3), [round(r, 3) for r in runs]


def gpu(fn, stream):
    """Median GPU-side duration (ms) of fn() enqueued on `stream`."""
    runs = []
    for i in range(WARMUP + REPS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            a.record()
            fn()
            b.record()
        b.synchronize()
        if i >= WARMUP:
            runs.append(a.elapsed_time(b))
    return round(statistics.median(runs), 3)


def link_bandwidth(nbytes, stream):
    out = {}
    dev = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    for pinned in (True, False):
        host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=pinned)
        h2d = gpu(lambda: dev.copy_(host, non_blocking=True), stream)
        d2h = gpu(lambda: host.copy_(dev, non_blocking=True), stream)
        out["pinned" if pinned else "pageable"] = {
            "h2d_gbps": round(nbytes / (h2d * 1e-3) / 1e9, 2),
            "d2h_gbps": round(nbytes / (d2h * 1e-3) / 1e9, 2),
        }
    return out


def two_streams(host, dev, gemm_iters, s_copy, s_comp):
    a = torch.randn(HIDDEN, HIDDEN, dtype=torch.float16, device="cuda")
    b = torch.randn(HIDDEN, HIDDEN, dtype=torch.float16, device="cuda")
    c = torch.empty_like(a)

    def copy():
        dev.copy_(host, non_blocking=True)

    def compute():
        for _ in range(gemm_iters):
            torch.mm(a, b, out=c)

    def one_stream():
        with torch.cuda.stream(s_comp):
            copy()
            compute()

    def both():
        with torch.cuda.stream(s_copy):
            copy()
        with torch.cuda.stream(s_comp):
            compute()

    res = {"copy_ms": gpu(copy, s_copy), "compute_ms": gpu(compute, s_comp)}
    res["one_stream_ms"], _ = wall(one_stream)
    res["two_streams_ms"], _ = wall(both)
    hidden = res["one_stream_ms"] - res["two_streams_ms"]
    res["hidden_pct_of_shorter"] = round(hidden / min(res["copy_ms"], res["compute_ms"]) * 100, 1)
    return res


def pipeline(host, x, y, w, chunks, s_copy, s_comp):
    rows = ROWS // chunks

    def run():
        for i in range(chunks):
            sl = slice(i * rows, (i + 1) * rows)
            with torch.cuda.stream(s_copy):
                x[sl].copy_(host[sl], non_blocking=True)
                ready = torch.cuda.Event()
                ready.record(s_copy)
            with torch.cuda.stream(s_comp):
                s_comp.wait_event(ready)
                torch.mm(x[sl], w, out=y[sl])

    def pingpong():
        # no cross-stream events: chunk i goes to stream i % 2, and stream
        # order alone keeps each chunk's GEMM after its own copy
        for i in range(chunks):
            sl = slice(i * rows, (i + 1) * rows)
            with torch.cuda.stream(s_copy if i % 2 else s_comp):
                x[sl].copy_(host[sl], non_blocking=True)
                torch.mm(x[sl], w, out=y[sl])

    def copies_first():
        # same dependencies as run(), but every copy is enqueued before any GEMM
        ready = []
        for i in range(chunks):
            sl = slice(i * rows, (i + 1) * rows)
            with torch.cuda.stream(s_copy):
                x[sl].copy_(host[sl], non_blocking=True)
                e = torch.cuda.Event()
                e.record(s_copy)
                ready.append(e)
        for i in range(chunks):
            sl = slice(i * rows, (i + 1) * rows)
            with torch.cuda.stream(s_comp):
                s_comp.wait_event(ready[i])
                torch.mm(x[sl], w, out=y[sl])

    def run_query():
        # run() plus a non-blocking stream.query() after each enqueue; on
        # WDDM-backed drivers a query is said to flush batched submissions
        for i in range(chunks):
            sl = slice(i * rows, (i + 1) * rows)
            with torch.cuda.stream(s_copy):
                x[sl].copy_(host[sl], non_blocking=True)
                ready = torch.cuda.Event()
                ready.record(s_copy)
            s_copy.query()
            with torch.cuda.stream(s_comp):
                s_comp.wait_event(ready)
                torch.mm(x[sl], w, out=y[sl])
            s_comp.query()

    ms, runs = wall(run)
    pp_ms, pp_runs = wall(pingpong)
    cf_ms, cf_runs = wall(copies_first)
    q_ms, q_runs = wall(run_query)
    return {"chunks": chunks, "ms": ms, "runs": runs, "pingpong_ms": pp_ms, "pingpong_runs": pp_runs,
            "copies_first_ms": cf_ms, "copies_first_runs": cf_runs,
            "query_ms": q_ms, "query_runs": q_runs}


def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else "0"
    s_copy, s_comp = torch.cuda.Stream(), torch.cuda.Stream()
    out = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "run": run_id,
           "rows": ROWS, "hidden": HIDDEN, "bytes_moved": ROWS * HIDDEN * 2,
           "warmup": WARMUP, "reps": REPS}

    out["link"] = link_bandwidth(ROWS * HIDDEN * 2, s_copy)
    print(json.dumps(out["link"]), flush=True)

    # one set of buffers, reused by every measurement below
    pinned = torch.randn(ROWS, HIDDEN, dtype=torch.float16).pin_memory()
    pageable = torch.randn(ROWS, HIDDEN, dtype=torch.float16)
    x = torch.empty(ROWS, HIDDEN, dtype=torch.float16, device="cuda")
    y = torch.empty_like(x)
    w = torch.randn(HIDDEN, HIDDEN, dtype=torch.float16, device="cuda")

    # size the GEMM loop so compute takes about as long as the pinned copy
    a = torch.randn(HIDDEN, HIDDEN, dtype=torch.float16, device="cuda")
    one_gemm = gpu(lambda: torch.mm(a, a), s_comp)
    copy_ms = gpu(lambda: x.copy_(pinned, non_blocking=True), s_copy)
    iters = max(1, round(copy_ms / one_gemm))
    out["gemm_4096_ms"] = one_gemm
    out["gemm_tflops"] = round(2 * HIDDEN**3 / (one_gemm * 1e-3) / 1e12, 1)
    out["gemm_iters"] = iters

    out["two_streams"] = {
        "pinned": two_streams(pinned, x, iters, s_copy, s_comp),
        "pageable": two_streams(pageable, x, iters, s_copy, s_comp),
    }
    print(json.dumps(out["two_streams"]), flush=True)

    copy_ms = gpu(lambda: x.copy_(pinned, non_blocking=True), s_copy)
    gemm_ms = gpu(lambda: torch.mm(x, w, out=y), s_comp)
    rows_out = [pipeline(pinned, x, y, w, c, s_copy, s_comp) for c in CHUNKS]
    for r in rows_out:
        r["ideal_ms"] = round(max(copy_ms, gemm_ms) + min(copy_ms, gemm_ms) / r["chunks"], 3)
    ref = torch.mm(pinned.cuda(), w)
    out["pipeline"] = {
        "copy_ms": copy_ms, "gemm_ms": gemm_ms, "rows": rows_out,
        "max_rel_diff_vs_serial": (
            (y.float() - ref.float()).abs().max() / ref.float().abs().max()).item(),
    }
    print(json.dumps({k: v for k, v in out["pipeline"].items() if k != "rows"}), flush=True)
    shown = ("chunks", "ms", "pingpong_ms", "copies_first_ms", "query_ms", "ideal_ms")
    for r in rows_out:
        print(json.dumps({k: r[k] for k in shown}), flush=True)

    os.makedirs("results", exist_ok=True)
    tag = out["gpu"].replace("NVIDIA GeForce ", "").replace(" ", "_").lower()
    with open(f"results/overlap_{tag}_run{run_id}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
