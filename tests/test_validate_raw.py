from pathlib import Path
import mne
from nai.product.validate_raw import validate_raw_object, validate_raw_path

BDF = Path(
    r"data/raw/ds006780/sub-10025/eeg/sub-10025_task-Restingstate_run-01_eeg.bdf"
)

def test_path_exists_or_skip():
    if not BDF.exists():
        print("SKIP: no BDF at", BDF)
        return
    r = validate_raw_path(BDF)
    assert r.status == "PASS", (r.reason, r.details)
    print("validate_raw_path: PASS", r.details)

def test_bad_path():
    r = validate_raw_path("does_not_exist.bdf")
    assert r.status == "REJECT"
    assert r.reason == "UNSUPPORTED_FORMAT"
    print("missing file: PASS")

if __name__ == "__main__":
    test_bad_path()
    test_path_exists_or_skip()
    print("P2.2 raw tests: done")