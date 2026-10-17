"""Annotation guideline shared by export_sheet.py (sheet "Guideline") and llm_label.py (prompt)."""

GUIDELINE = (
    "For each row, judge the sentiment the writer expresses toward the [[target]] only, not the "
    "sentence overall. positive = the writer expresses a favourable opinion of the target. "
    "negative = unfavourable opinion. neutral = no clear positive or negative opinion about the "
    "target (mentioning, describing or listing it without evaluating it). Use only the sentence. "
    "Mark sure=no if you could defend a different label."
)

LABELS = ["positive", "negative", "neutral"]
SHEET = "Annotation"
DATASETS = {
    "restaurant": ("data/processed/restaurant14_test_implicit.json", "data/processed/restaurant14_test_all.json"),
    "laptop": ("data/processed/laptop14_test_implicit.json", "data/processed/laptop14_test_all.json"),
}
