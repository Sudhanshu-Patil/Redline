"""Unit tests for the scanned-PDF adapter's pure logic: TSV line grouping and
the vision-fallback policy. No tesseract binary or network required."""

import pytest

from src.canonical.model import BBox, Element
from src.config import settings
from src.ingest.pdf_scanned import (
    OcrLine,
    PdfScannedAdapter,
    clean_ocr_line_text,
    cluster_lines_spatially,
    group_tsv_into_lines,
    resolve_tesseract_cmd,
)


class TestResolveTesseractCmd:
    def test_explicit_config_wins(self, monkeypatch):
        monkeypatch.setattr(settings, "tesseract_cmd", r"C:\custom\tesseract.exe")
        assert resolve_tesseract_cmd() == r"C:\custom\tesseract.exe"

    def test_path_lookup_used_when_no_config(self, monkeypatch):
        monkeypatch.setattr(settings, "tesseract_cmd", "")
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/tesseract")
        assert resolve_tesseract_cmd() == "/usr/bin/tesseract"

    def test_none_when_nothing_found(self, monkeypatch):
        monkeypatch.setattr(settings, "tesseract_cmd", "")
        monkeypatch.setattr("shutil.which", lambda _: None)
        monkeypatch.setattr(
            "src.ingest.pdf_scanned._WINDOWS_TESSERACT_DEFAULTS", []
        )
        assert resolve_tesseract_cmd() is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("PIT \\", "PIT"),  # bubble edge glued as separate token
        ("( TIT", "TIT"),
        ("{ PIT \\", "PIT"),
        ("9062 /", "9062"),
        ("(TIT", "TIT"),  # junk fused to the token itself
        ("9062/", "9062"),
        ("SP = 260 bar (g)", "SP = 260 bar (g"),  # documented cosmetic loss only
        ("30.", "30."),  # note markers keep their period
        ('4"x6"', '4"x6"'),  # size quotes preserved
        ('6"-VF-43-9029-AC21S-00', '6"-VF-43-9029-AC21S-00'),
        ("NOTE 31", "NOTE 31"),
        ("\\ /", ""),  # pure line-art noise vanishes
        ("*", ""),
    ],
)
def test_clean_ocr_line_text(raw, expected):
    assert clean_ocr_line_text(raw) == expected


def test_cleaned_bubble_lines_classify_as_instrument_loop():
    """End-to-end through the shared classifier: noisy OCR bubble text must
    still produce an instrument_loop element after cleaning."""
    from src.ingest.pdf_native import classify_block_lines

    cleaned = [clean_ocr_line_text(t) for t in ["PIT \\", "9062 /"]]
    drafts = classify_block_lines([(t, (0.0, 0.0, 1.0, 1.0)) for t in cleaned])
    assert len(drafts) == 1
    assert drafts[0].type == "instrument_loop"
    assert drafts[0].text == "PIT-9062"


def line(text, x0, y0, x1, y1, block=1, key=(1, 1, 1), conf=90.0):
    return OcrLine(
        block_num=block, line_key=key, text=text, x0=x0, y0=y0, x1=x1, y1=y1, confidence=conf
    )


class TestClusterLinesSpatially:
    def test_instrument_bubble_stack_clusters_across_tesseract_blocks(self):
        """PIT / 9062 in separate tesseract blocks (typical under PSM 11) must
        still land in one spatial group so classify_block_lines can merge them."""
        pit = line("PIT", 100, 100, 130, 123, block=1, key=(1, 1, 1))
        loop = line("9062", 95, 128, 143, 151, block=2, key=(2, 1, 1))
        far_away = line("VENDOR", 800, 100, 900, 123, block=3, key=(3, 1, 1))
        groups = cluster_lines_spatially([far_away, loop, pit])
        texts = [[ln.text for ln in g] for g in groups]
        assert [["PIT", "9062"], ["VENDOR"]] == texts

    def test_horizontally_disjoint_stacks_stay_separate(self):
        a1 = line("HH: 250", 200, 100, 260, 123)
        b1 = line("LL: 120", 400, 100, 460, 123)  # same rows, far right
        a2 = line("RS", 205, 128, 225, 151)
        groups = cluster_lines_spatially([a1, b1, a2])
        texts = sorted([sorted(ln.text for ln in g) for g in groups])
        assert [["HH: 250", "RS"], ["LL: 120"]] == texts

    def test_large_vertical_gap_breaks_cluster(self):
        top = line("NOTES:", 100, 100, 160, 123)
        bottom = line("1. FIRST NOTE", 100, 400, 260, 423)
        groups = cluster_lines_spatially([top, bottom])
        assert len(groups) == 2

    def test_note_column_forms_one_group(self):
        rows = [
            line(f"{i}. NOTE TEXT {i}", 100, 100 + i * 24, 300, 100 + i * 24 + 22)
            for i in range(1, 6)
        ]
        groups = cluster_lines_spatially(rows)
        assert len(groups) == 1
        assert [ln.text for ln in groups[0]] == [f"{i}. NOTE TEXT {i}" for i in range(1, 6)]

    def test_empty_input(self):
        assert cluster_lines_spatially([]) == []


def make_tsv(rows):
    """rows: list of (block, par, line, word, left, top, width, height, conf, text)."""
    keys = [
        "block_num", "par_num", "line_num", "word_num",
        "left", "top", "width", "height", "conf", "text",
    ]  # fmt: skip
    return {k: [r[i] for r in rows] for i, k in enumerate(keys)}


class TestGroupTsvIntoLines:
    def test_words_on_same_line_are_joined_in_order(self):
        tsv = make_tsv(
            [
                (1, 1, 1, 1, 10, 10, 30, 12, 90.0, "SP"),
                (1, 1, 1, 2, 45, 10, 10, 12, 85.0, "="),
                (1, 1, 1, 3, 60, 10, 30, 12, 95.0, "260"),
            ]
        )
        lines = group_tsv_into_lines(tsv)
        assert len(lines) == 1
        assert lines[0].text == "SP = 260"
        assert lines[0].confidence == (90.0 + 85.0 + 95.0) / 3

    def test_line_bbox_is_union_of_word_boxes(self):
        tsv = make_tsv(
            [
                (1, 1, 1, 1, 10, 10, 30, 12, 90.0, "HH:"),
                (1, 1, 1, 2, 45, 8, 20, 16, 90.0, "250"),
            ]
        )
        line = group_tsv_into_lines(tsv)[0]
        assert (line.x0, line.y0, line.x1, line.y1) == (10.0, 8.0, 65.0, 24.0)

    def test_structural_rows_and_empty_text_are_dropped(self):
        tsv = make_tsv(
            [
                (1, 1, 1, 0, 0, 0, 100, 100, -1.0, ""),  # structural row
                (1, 1, 1, 1, 10, 10, 30, 12, 90.0, "PIT"),
                (1, 1, 1, 2, 45, 10, 10, 12, 88.0, "   "),  # whitespace only
            ]
        )
        lines = group_tsv_into_lines(tsv)
        assert len(lines) == 1
        assert lines[0].text == "PIT"

    def test_separate_lines_and_blocks_stay_separate(self):
        tsv = make_tsv(
            [
                (1, 1, 1, 1, 10, 10, 30, 12, 90.0, "PIT"),
                (1, 1, 2, 1, 10, 25, 30, 12, 92.0, "9062"),
                (2, 1, 1, 1, 200, 10, 30, 12, 80.0, "VENDOR"),
            ]
        )
        lines = group_tsv_into_lines(tsv)
        assert [ln.text for ln in lines] == ["PIT", "9062", "VENDOR"]
        assert lines[0].block_num == 1 and lines[2].block_num == 2

    def test_empty_tsv_yields_no_lines(self):
        assert group_tsv_into_lines(make_tsv([])) == []


def _element(text: str, conf: float, seq: int = 0) -> Element:
    return Element(
        id=f"t:pdf_scanned:{seq:05d}",
        type="text_block",
        text=text,
        bbox=BBox(
            page=0, x0=100, y0=100, x1=200, y1=120, page_width=1191, page_height=842
        ),
        source_adapter="pdf_scanned",
        extraction_confidence=conf,
    )


class FakeVision:
    def __init__(self, reply: str, configured: bool = True):
        self.reply = reply
        self.configured = configured
        self.calls: list[str] = []

    @property
    def is_configured(self) -> bool:
        return self.configured

    def read_image_text(self, png_bytes: bytes, context_hint: str = "") -> str:
        self.calls.append(context_hint)
        return self.reply


class TestVisionFallback:
    def _run(self, elements, vision, monkeypatch=None):
        from PIL import Image

        adapter = PdfScannedAdapter(vision_client=vision)
        image = Image.new("RGB", (1000, 800), "white")
        scale = 72.0 / settings.ocr_dpi
        adapter._apply_vision_fallback(elements, image, scale, page_index=0)

    def test_low_confidence_element_gets_replaced_and_original_kept(self):
        vision = FakeVision(reply="SP = 260 bar (g)")
        low = _element("5P = 26O bar (g)", conf=0.3)
        self._run([low], vision)
        assert low.text == "SP = 260 bar (g)"
        assert low.attributes["ocr_text"] == "5P = 26O bar (g)"
        assert low.attributes["ocr_fallback"] == "vision_llm"
        assert low.extraction_confidence == settings.vision_fallback_confidence
        assert len(vision.calls) == 1

    def test_high_confidence_element_untouched(self):
        vision = FakeVision(reply="SHOULD NOT BE USED")
        high = _element("PIT", conf=0.95)
        self._run([high], vision)
        assert high.text == "PIT"
        assert "ocr_fallback" not in high.attributes
        assert vision.calls == []

    def test_llm_agreeing_with_ocr_boosts_confidence_without_rewrite(self):
        vision = FakeVision(reply="HH: 250")
        low = _element("HH: 250", conf=0.4)
        self._run([low], vision)
        assert low.text == "HH: 250"
        assert "ocr_text" not in low.attributes
        assert low.attributes["ocr_fallback"] == "vision_llm_confirmed"
        assert low.extraction_confidence == settings.vision_fallback_confidence

    def test_unconfigured_client_marks_elements_and_makes_no_calls(self):
        vision = FakeVision(reply="X", configured=False)
        low = _element("blurry text", conf=0.2)
        self._run([low], vision)
        assert low.text == "blurry text"
        assert low.attributes["ocr_fallback"] == "unavailable"
        assert vision.calls == []

    def test_region_cap_prioritizes_lowest_confidence(self, monkeypatch):
        monkeypatch.setattr(settings, "vision_fallback_max_regions", 2)
        vision = FakeVision(reply="")
        worst = _element("worst", conf=0.1, seq=1)
        middle = _element("middle", conf=0.3, seq=2)
        best_of_bad = _element("best-of-bad", conf=0.5, seq=3)
        self._run([best_of_bad, worst, middle], vision)
        assert len(vision.calls) == 2
        assert "worst" in vision.calls[0]
        assert "middle" in vision.calls[1]
        assert "ocr_fallback" not in best_of_bad.attributes

    def test_single_character_noise_not_sent_to_llm(self):
        vision = FakeVision(reply="X")
        noise = _element("*", conf=0.1)
        self._run([noise], vision)
        assert vision.calls == []

    def test_vision_call_failure_marks_element_and_continues(self):
        class ExplodingVision:
            @property
            def is_configured(self):
                return True

            def read_image_text(self, png_bytes, context_hint=""):
                raise ConnectionError("network down")

        low = _element("readable text", conf=0.2)
        self._run([low], ExplodingVision())
        assert low.text == "readable text"  # original kept
        assert low.attributes["ocr_fallback"] == "error"


def test_ingest_raises_clear_error_when_tesseract_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "tesseract_cmd", "")
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setattr("src.ingest.pdf_scanned._WINDOWS_TESSERACT_DEFAULTS", [])
    with pytest.raises(RuntimeError, match="tesseract"):
        PdfScannedAdapter().ingest(tmp_path / "any.pdf", pid="x")
