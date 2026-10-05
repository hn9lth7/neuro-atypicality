from nai.product.report import build_reject_report, build_pass_report
from nai.product.model_bundle import load_bundle

def main() -> None:
    r = build_reject_report(reason="MISSING_AGE", details={"field": "age"})
    assert r["status"] == "REJECT" and r["scores"] is None

    bundle = load_bundle("models/nai_v1")
    r = build_pass_report(
        scores={
            "D_SE": 1.0,
            "D_C": 1.0,
            "D_G": 1.0,
            "D_D": 1.0,
            "NAI": 1.0,
        },
        bundle=bundle,
        participant_id="sub-test",
        age=10.0,
    )
    assert r["status"] == "PASS" and r["scores"]["NAI"] == 1.0
    print("P3 report tests: PASS")

if __name__ == "__main__":
    main()