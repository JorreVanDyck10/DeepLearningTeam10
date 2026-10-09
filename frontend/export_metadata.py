"""Refresh UI options and metrics from the trusted local baseline. Run from repo root."""
import json
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "secondary_mushroom/models/baseline_decision_tree.joblib"
model = joblib.load(MODEL)
colors = dict(n="Bruin", b="Beige", g="Grijs", r="Groen", p="Roze", u="Paars", e="Rood", w="Wit", y="Geel", l="Blauw", o="Oranje", k="Zwart", f="Geen")
surfaces = dict(i="Vezelig", g="Gegroefd", y="Schubbig", s="Glad", h="Glanzend", l="Leerachtig", k="Zijdeachtig", t="Kleverig", w="Gerimpeld", e="Vlezig", d="Droog", f="Geen")
labels = {
    "cap-diameter": ("Hoeddiameter", "cm"), "stem-height": ("Steelhoogte", "cm"),
    "stem-width": ("Steeldikte", "mm"), "cap-shape": ("Hoedvorm", ""),
    "cap-color": ("Hoedkleur", ""), "season": ("Seizoen", ""),
    "cap-surface": ("Hoedoppervlak", ""), "does-bruise-or-bleed": ("Verkleurt of bloedt", ""),
    "gill-attachment": ("Lamelaanhechting", ""), "gill-spacing": ("Afstand tussen lamellen", ""),
    "gill-color": ("Lamelkleur", ""), "stem-root": ("Steelbasis", ""),
    "stem-surface": ("Steeloppervlak", ""), "stem-color": ("Steelkleur", ""),
    "veil-type": ("Type sluier", ""), "veil-color": ("Sluierkleur", ""),
    "has-ring": ("Ring aanwezig", ""), "ring-type": ("Ringtype", ""),
    "spore-print-color": ("Kleur sporenafdruk", ""), "habitat": ("Leefomgeving", ""),
}
maps = {
    "cap-shape": dict(b="Klokvormig", c="Kegelvormig", x="Bol", f="Vlak", s="Ingedrukt", p="Bolvormig", o="Anders"),
    "cap-surface": surfaces, "stem-surface": surfaces,
    "does-bruise-or-bleed": dict(t="Ja", f="Nee"), "has-ring": dict(t="Ja", f="Nee"),
    "gill-attachment": dict(a="Breed aangehecht", x="Smal aangehecht", d="Aflopend", e="Vrij", s="Uitgebocht", p="Poriën", f="Geen", **{"?": "Onbekend (datasetcode)"}),
    "gill-spacing": dict(c="Dicht bij elkaar", d="Ver uit elkaar", f="Geen"),
    "stem-root": dict(b="Knolvormig", s="Gezwollen", c="Knotsvormig", u="Bekervormig", e="Gelijkmatig", z="Wortelstrengen", r="Wortelend", f="Geen"),
    "veil-type": dict(p="Gedeeltelijk", u="Algemeen"),
    "ring-type": dict(c="Spinnenwebachtig", e="Vergankelijk", r="Uitstaand", g="Gegroefd", l="Groot", p="Hangend", s="Schedevormig", z="Zone", y="Schubbig", m="Beweegbaar", f="Geen", **{"?": "Onbekend (datasetcode)"}),
    "habitat": dict(g="Gras", l="Bladeren", m="Weide", p="Paden", h="Heide", u="Stedelijk", w="Afvalterrein", d="Bos"),
    "season": dict(s="Lente", u="Zomer", a="Herfst", w="Winter"),
}
for name in ["cap-color", "gill-color", "stem-color", "veil-color", "spore-print-color"]:
    maps[name] = colors
categories = {}
for name, transformer, columns in model.steps[0][1].transformers_:
    if hasattr(transformer, "steps") and hasattr(transformer.steps[-1][1], "categories_"):
        categories.update(zip(columns, transformer.steps[-1][1].categories_))
order = ["cap-diameter", "cap-shape", "cap-color", "stem-height", "stem-width", "season"]
order += [name for name in model.feature_names_in_ if name not in order]
fields = []
for name in order:
    label, unit = labels[name]
    field = {"name": name, "label": label, "unit": unit}
    if name in categories:
        field["options"] = [[str(code), maps[name].get(str(code), f"Datasetcode {code}")] for code in categories[name] if isinstance(code, str)]
    fields.append(field)
metrics = json.loads((ROOT / "secondary_mushroom/metrics/baseline_metrics.json").read_text())
example = json.loads((ROOT / "secondary_mushroom/example_mushroom.json").read_text())[0]
data = {"fields": fields, "example": example, "metrics": metrics["results"]["decision_tree_depth_5"]}
Path(__file__).with_name("metadata.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("Exported 20 baseline fields, supported category codes, example and real metrics.")
