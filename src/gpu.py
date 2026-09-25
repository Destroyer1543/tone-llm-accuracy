import subprocess
import time
import os


def query_gpus():
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    ).decode()
    gpus = []
    for line in out.strip().splitlines():
        idx, free, used, util = [x.strip() for x in line.split(",")]
        gpus.append(
            {
                "index": int(idx),
                "mem_free_mb": int(free),
                "mem_used_mb": int(used),
                "util_pct": int(util),
            }
        )
    return gpus


def _allowed():
    raw = os.environ.get("AAAI_ALLOWED_GPUS", "0,1")
    return {int(x) for x in raw.split(",") if x.strip() != ""}


def pick_gpu(min_free_mb=20000, wait=True, poll_sec=30, timeout_sec=None):
    allowed = _allowed()
    start = time.time()
    while True:
        gpus = [g for g in query_gpus() if g["index"] in allowed]
        gpus.sort(key=lambda g: g["mem_free_mb"], reverse=True)
        best = gpus[0]
        if best["mem_free_mb"] >= min_free_mb or not wait:
            return best["index"]
        if timeout_sec is not None and (time.time() - start) > timeout_sec:
            raise TimeoutError(
                f"No GPU with {min_free_mb}MB free after {timeout_sec}s. "
                f"State: {gpus}"
            )
        print(
            f"[gpu] waiting for {min_free_mb}MB free; best is GPU{best['index']} "
            f"with {best['mem_free_mb']}MB. Repolling in {poll_sec}s...",
            flush=True,
        )
        time.sleep(poll_sec)


def set_visible_gpu(min_free_mb=20000, wait=True):
    idx = pick_gpu(min_free_mb=min_free_mb, wait=wait)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    print(f"[gpu] pinned to GPU {idx} (CUDA_VISIBLE_DEVICES={idx})", flush=True)
    return idx


if __name__ == "__main__":
    for g in query_gpus():
        print(g)
    print("would pick:", pick_gpu(wait=False))
