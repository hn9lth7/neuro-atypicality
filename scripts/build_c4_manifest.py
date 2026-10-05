import json
from pathlib import Path
from datetime import datetime

# Дані з попереднього аудиту
manifest = {
    "cohort_id": "sheffield_orda_dickinson_2022",
    "independent_of_ds006780": True,
    "n_asd": 28,
    "n_td": 19,
    "age_min": 18,
    "age_max": 68,
    "resting": True,
    "eyes_condition": "closed",
    "raw_access": "yes",
    "srate_hz": 512,
    "n_channels": 63,
    "montage": "BioSemi 64 (Cz excluded as reference)",
    "reference": "common",
    "labels_subject_level": True,
    "overlap_with_development": False,
    "legal_access": True,
    # Додаткові поля для документації
    "source": "ORDA/Figshare DOI 10.15131/shef.data.16840351",
    "paper": "Dickinson, Jeste & Milne (2022), PMID 35176551",
    "duration_sec": 150,
    "interpolation_method": "MNE spherical spline",
    "mean_channels_interpolated": 5.09,
    "max_channels_interpolated": 14,
    "n_subjects_total": 47,
    "excluded_alt_montage": 9,
    "generated": datetime.utcnow().isoformat() + "Z",
}

out = Path("data/external/sheffield_asd_metadata/sheffield_orda_c4_manifest.json")
out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Saved -> {out}")
print(json.dumps(manifest, indent=2))
