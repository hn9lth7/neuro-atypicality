from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_KEYS = [
    "cohort_id",
    "independent_of_ds006780",
    "n_asd",
    "n_td",
    "age_min",
    "age_max",
    "resting",
    "eyes_condition",  
    "raw_access",     
    "srate_hz",
    "n_channels",
    "montage",
    "reference",
    "labels_subject_level",
    "overlap_with_development",  
    "legal_access",
]

def evaluate(m: dict) -> dict:
    issues, warnings = [], []
    def need(k, pred, msg):
        if not pred:
            issues.append(msg)

    for k in REQUIRED_KEYS:
        if k not in m:
            issues.append(f"missing manifest key: {k}")

    if m.get("independent_of_ds006780") is not True:
        issues.append("not independent of ds006780")
    if m.get("overlap_with_development") is True:
        issues.append("subject overlap with development set")
    if m.get("raw_access") != "yes":
        issues.append(f"raw_access={m.get('raw_access')} (need yes for C4-C)")
    if not m.get("resting"):
        issues.append("not resting EEG")
    if not m.get("labels_subject_level"):
        issues.append("labels not subject-level")
    if not m.get("legal_access"):
        issues.append("legal access not confirmed")
    n_asd, n_td = m.get("n_asd") or 0, m.get("n_td") or 0
    if n_asd < 1 or n_td < 1:
        issues.append("need both ASD and TD/NT counts ≥ 1")

    n_ch = m.get("n_channels")
    if n_ch is not None and n_ch < 64:
        warnings.append(
            f"n_channels={n_ch}: frozen 64-ch 54-D semantics likely incompatible; "
            "requires reduced-montage contract before extraction"
        )
    srate = m.get("srate_hz")
    if srate is not None and srate < 128:
        warnings.append(f"srate={srate} may be insufficient for gamma/PLV as defined")

    status = "NO-GO" if issues else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    return {"status": status, "issues": issues, "warnings": warnings, "manifest": m}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    args = p.parse_args()
    m = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rep = evaluate(m)
    out = Path("results/ml") / f"c4_{m.get('cohort_id','cohort')}_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    print(f"Saved → {out}")

if __name__ == "__main__":
    main()