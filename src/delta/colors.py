"""Shared change-type color/label convention (plan §5, §8): delta/report.py's
charts and markup/overlay.py's bounding-box highlights both draw from this
one definition, so the HTML report and the annotated drawing always agree
on what green/amber/red mean. Split out to a standalone module (rather than
living in report.py, where it started) because overlay.py needs it too and
report.py importing overlay.py (for the markup preview embed) would
otherwise create a circular import.
"""

STATUS_COLORS = {
    "added": "#0ca30c",  # good
    "modified": "#fab219",  # warning
    "removed": "#d03b3b",  # critical
}
STATUS_LABELS = {"added": "Added", "modified": "Modified", "removed": "Removed"}
CHANGE_ORDER = ["added", "modified", "removed"]
