from dataclasses import dataclass
from typing import Any, Dict


ORDER = ["Normal", "Benign", "Malignant"]


def _canon_label(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    low = s.lower()
    if low == "bengin":
        return "Benign"
    if low in ("normal", "benign", "malignant"):
        return low.capitalize()
    return s


def _to_pct_confidence(x: Any) -> int:
    try:
        v = float(x)
        # backend sometimes returns 0..1, sometimes 0..100
        if 0 <= v <= 1:
            v *= 100.0
        return max(0, min(100, int(round(v))))
    except Exception:
        return 0


def _normalize_probs_to_100(raw: Dict[str, Any]) -> Dict[str, int]:
    # raw can be: softmax (sum~1), percents (sum~100), or OVR scores (sum>100)
    vals: Dict[str, float] = {}
    for k, v in (raw or {}).items():
        kk = _canon_label(k)
        if kk in ORDER:
            try:
                vals[kk] = max(0.0, float(v))
            except Exception:
                vals[kk] = 0.0

    for k in ORDER:
        vals.setdefault(k, 0.0)

    s = sum(vals.values())
    if s <= 0:
        return {k: 0 for k in ORDER}

    # Largest remainder rounding to guarantee sum==100
    floats = {k: (vals[k] / s) * 100.0 for k in ORDER}
    floors = {k: int(floats[k]) for k in ORDER}
    remainder = 100 - sum(floors.values())

    fracs = sorted(((floats[k] - floors[k], k) for k in ORDER), reverse=True)
    out = dict(floors)
    for i in range(remainder):
        out[fracs[i][1]] += 1

    return out


@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence_pct: int
    probabilities_pct: Dict[str, int]


def parse_backend(data: Dict[str, Any]) -> PredictionResult:
    label = _canon_label(data.get("label"))
    confidence_pct = _to_pct_confidence(data.get("confidence"))
    probs_pct = _normalize_probs_to_100(data.get("all_probabilities") or {})

    # fallback: if confidence missing, take max probability
    if confidence_pct == 0 and any(probs_pct.values()):
        confidence_pct = max(probs_pct.values())

    return PredictionResult(
        label=label,
        confidence_pct=confidence_pct,
        probabilities_pct=probs_pct,
    )
