"""Median (min-max) across results/overlap_*_run*.json -> results/summary.json."""
import glob
import json
import statistics

runs = [json.load(open(f)) for f in sorted(glob.glob("results/overlap_*_run*.json"))]


def agg(values):
    return {"median": round(statistics.median(values), 2), "min": round(min(values), 2),
            "max": round(max(values), 2), "n": len(values)}


out = {"gpu": runs[0]["gpu"], "runs": len(runs), "link": {}, "two_streams": {}, "pipeline": []}
for kind in ("pinned", "pageable"):
    out["link"][kind] = {d: agg([r["link"][kind][d] for r in runs]) for d in ("h2d_gbps", "d2h_gbps")}
    out["two_streams"][kind] = {
        k: agg([r["two_streams"][kind][k] for r in runs])
        for k in ("copy_ms", "compute_ms", "one_stream_ms", "two_streams_ms", "hidden_pct_of_shorter")}
out["pipeline_parts"] = {k: agg([r["pipeline"][k] for r in runs]) for k in ("copy_ms", "gemm_ms")}
for i, row in enumerate(runs[0]["pipeline"]["rows"]):
    out["pipeline"].append({"chunks": row["chunks"], **{
        k: agg([r["pipeline"]["rows"][i][k] for r in runs])
        for k in ("ms", "pingpong_ms", "copies_first_ms", "query_ms", "ideal_ms")}})

json.dump(out, open("results/summary.json", "w"), indent=2)
print(json.dumps(out, indent=1))
