import time
import requests
from .schemas import PredictionResult, parse_backend

class BackendError(RuntimeError):
    pass

def predict(api_url: str, file_bytes: bytes, filename: str, timeout_s: int = 20) -> tuple[PredictionResult, int]:
    t0 = time.time()
    files = {"image": (filename, file_bytes, "application/octet-stream")}
    r = requests.post(api_url, files=files, timeout=timeout_s)
    latency_ms = int((time.time() - t0) * 1000)

    if r.status_code != 200:
        raise BackendError(f"HTTP {r.status_code}: {r.text}")

    return parse_backend(r.json()), latency_ms
