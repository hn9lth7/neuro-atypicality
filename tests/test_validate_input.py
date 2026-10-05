from nai.product.validate_input import validate_feature_row
from nai.features.blocks import ALL_BLOCKS

def valid_row():
    row = {"age": 10.0}
    for features in ALL_BLOCKS.values():
        for feature in features:
            row[feature] = 0.0
    return row

r = validate_feature_row(valid_row())
assert r.status == "PASS" and r.reason is None and r.warnings == []

row = valid_row()
del row["alpha_rel"]
r = validate_feature_row(row)
assert r.status == "REJECT" and r.reason == "MISSING_FEATURES"

row = valid_row()
row["alpha_rel"] = float("nan")
r = validate_feature_row(row)
assert r.status == "REJECT" and r.reason == "INVALID_FEATURES"

row = valid_row()
del row["age"]
r = validate_feature_row(row)
assert r.status == "REJECT" and r.reason == "MISSING_AGE"

row = valid_row()
row["age"] = 14.5
r = validate_feature_row(row)
assert r.status == "PASS"
assert any("AGE_OUT_OF_RANGE" in w for w in r.warnings)

print("P2 validation tests: PASS")