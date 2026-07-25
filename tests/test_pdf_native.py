from src.ingest.pdf_native import ElementDraft, classify_block_lines

B = (0.0, 0.0, 10.0, 10.0)  # placeholder bbox; classify_block_lines doesn't inspect its values


def L(text: str, bbox=B):
    return (text, bbox)


def test_element_draft_is_not_equal_to_a_non_element_draft():
    # __eq__ must return NotImplemented (not raise/crash) for a type
    # mismatch, the standard Python protocol for "let the other side try".
    draft = ElementDraft("note", "1. TEXT", B)
    assert draft != "1. TEXT"
    assert draft != 5


def test_element_draft_repr_includes_type_and_text():
    draft = ElementDraft("note", "1. TEXT", B)
    assert repr(draft) == "ElementDraft('note', '1. TEXT', attributes={})"


def test_note_single_line_body():
    lines = [L("1."), L("26-PDI-9054 HH INITIATE PRESSURIZED COMPRESSOR STOP.")]
    result = classify_block_lines(lines)
    assert result == [
        ElementDraft(
            "note",
            "1. 26-PDI-9054 HH INITIATE PRESSURIZED COMPRESSOR STOP.",
            B,
            {"kind": "definition", "note_number": "1"},
        )
    ]


def test_note_wraps_across_multiple_lines_until_next_marker():
    lines = [
        L("4."),
        L("HIGH POINT VENT AND LOW POINT DRAIN TO BE FINALIZED BY PIPING AS PER PIPING"),
        L("LAYOUT."),
        L("5."),
        L("OIL CHANGE."),
    ]
    result = classify_block_lines(lines)
    assert len(result) == 2
    assert result[0].type == "note"
    assert result[0].attributes["note_number"] == "4"
    assert "LAYOUT." in result[0].text
    assert result[1].attributes["note_number"] == "5"


def test_note_inline_marker_and_body_on_one_line():
    """Notes 16+ in the sample P&ID carry the marker and body on one physical
    line, unlike notes 1-15 which have the bare marker on its own line."""
    lines = [L("22. DESIGN PRESSURE IN EXTERNAL SYSTEM DOWNSTREAM COMPRESSOR 257 BARG.")]
    result = classify_block_lines(lines)
    assert result == [
        ElementDraft(
            "note",
            "22. DESIGN PRESSURE IN EXTERNAL SYSTEM DOWNSTREAM COMPRESSOR 257 BARG.",
            B,
            {"kind": "definition", "note_number": "22"},
        )
    ]


def test_note_inline_marker_with_wrapped_continuation():
    lines = [
        L("16. PRIMARY SEAL GAS IS TAKEN DOWNSTREAM LAST COMPRESSING STAGE (8TH STAGE). SKID"),
        L("INTERNAL PIPING TO BE INSULATED."),
        L("17. SECONDARY SEAL GAS AND SEPARATION GAS."),
    ]
    result = classify_block_lines(lines)
    assert len(result) == 2
    assert result[0].attributes["note_number"] == "16"
    assert result[0].text.endswith("SKID INTERNAL PIPING TO BE INSULATED.")
    expected_attrs = {"kind": "definition", "note_number": "17"}
    assert result[1] == ElementDraft(
        "note", "17. SECONDARY SEAL GAS AND SEPARATION GAS.", B, expected_attrs
    )


def test_note_range_marker():
    lines = [L("1-23."), L("DELETED."), L("HOLDS:")]
    result = classify_block_lines(lines)
    assert result[0] == ElementDraft(
        "note", "1-23. DELETED.", B, {"kind": "definition", "note_number": "1-23"}
    )
    # section header stops note consumption and falls through to generic text_block
    assert result[1] == ElementDraft("text_block", "HOLDS:", B)


def test_note_reference_callout():
    lines = [L("NOTE 13,21")]
    result = classify_block_lines(lines)
    assert result == [ElementDraft("note", "NOTE 13,21", B, {"kind": "reference", "refs": "13,21"})]


def test_line_number():
    lines = [L('6"-VF-43-9029-AC21S-00')]
    result = classify_block_lines(lines)
    assert result == [ElementDraft("line_number", '6"-VF-43-9029-AC21S-00', B)]


def test_valve_tag():
    lines = [L("26BL9072")]
    assert classify_block_lines(lines) == [ElementDraft("valve", "26BL9072", B)]


def test_equipment_tag():
    lines = [L("26-HA-911")]
    assert classify_block_lines(lines) == [ElementDraft("tag", "26-HA-911", B)]


def test_setpoint_inline_sp_equals():
    lines = [L("SP = 257 bar (g)")]
    result = classify_block_lines(lines)
    expected_attrs = {"setpoint_type": "SP", "value": "257", "unit": "bar (g)"}
    assert result == [ElementDraft("setpoint", "SP = 257 bar (g)", B, expected_attrs)]


def test_setpoint_set_pressure_variant():
    lines = [L("SET PRESSURE=10 bar g")]
    result = classify_block_lines(lines)
    assert result[0].type == "setpoint"
    assert result[0].attributes["value"] == "10"


def test_setpoint_hh_limit():
    lines = [L("HH: 245")]
    result = classify_block_lines(lines)
    expected_attrs = {"setpoint_type": "HH", "value": "245"}
    assert result == [ElementDraft("setpoint", "HH: 245", B, expected_attrs)]


def test_size_transition_dimension():
    lines = [L('4"x6"')]
    result = classify_block_lines(lines)
    assert result == [ElementDraft("dimension", '4"x6"', B, {"kind": "size_transition"})]


def test_instrument_loop_two_line_cluster():
    lines = [L("PIT"), L("9062")]
    result = classify_block_lines(lines)
    assert result == [
        ElementDraft(
            "instrument_loop", "PIT-9062", B, {"function": "PIT", "loop_number": "9062"}
        )
    ]


def test_instrument_loop_three_line_cluster_with_unit():
    lines = [L("PIT"), L("9055"), L("26")]
    result = classify_block_lines(lines)
    assert result == [
        ElementDraft(
            "instrument_loop",
            "PIT-9055",
            B,
            {"function": "PIT", "loop_number": "9055", "unit": "26"},
        )
    ]


def test_instrument_loop_with_parallel_unit_letter_clusters():
    """PSV bubbles carry lettered loop numbers (9066A/9066B) — these must
    cluster exactly like plain numeric loops."""
    lines = [L("PSV"), L("9066A")]
    result = classify_block_lines(lines)
    assert result == [
        ElementDraft(
            "instrument_loop", "PSV-9066A", B, {"function": "PSV", "loop_number": "9066A"}
        )
    ]


def test_hyphenated_instrument_tag_single_token():
    """DWG ATTRIBs and prose carry tags as one token: 'PIT-9062'."""
    result = classify_block_lines([L("PIT-9062")])
    assert result == [
        ElementDraft(
            "instrument_loop", "PIT-9062", B, {"function": "PIT", "loop_number": "9062"}
        )
    ]


def test_hyphenated_token_with_unknown_function_code_stays_text():
    result = classify_block_lines([L("ZZZ-9062")])
    assert result[0].type == "text_block"


def test_instrument_loop_rejects_unit_code_as_loop_number():
    """PI followed by a 2-digit unit code (not a 3-6 digit loop number) must NOT
    merge -- this is the guard against mis-clustering 'PI' + area-code '26'."""
    lines = [L("PI"), L("26")]
    result = classify_block_lines(lines)
    assert result == [
        ElementDraft("text_block", "PI", B),
        ElementDraft("text_block", "26", B),
    ]


def test_valve_status_flag():
    lines = [L("CSO")]
    assert classify_block_lines(lines) == [
        ElementDraft("text_block", "CSO", B, {"kind": "valve_status_flag"})
    ]


def test_single_letter_flag():
    lines = [L("U")]
    assert classify_block_lines(lines) == [ElementDraft("text_block", "U", B, {"kind": "flag"})]


def test_unclassified_text_falls_back_to_text_block():
    lines = [L("VENDOR")]
    assert classify_block_lines(lines) == [ElementDraft("text_block", "VENDOR", B)]


def test_asterisk_marker_is_not_dropped():
    lines = [L("*")]
    assert classify_block_lines(lines) == [ElementDraft("text_block", "*", B)]


def test_instrument_function_code_at_end_of_block_does_not_crash():
    """No line after the function code -- must fall through safely, not IndexError."""
    lines = [L("VENDOR"), L("PIT")]
    result = classify_block_lines(lines)
    assert result[-1] == ElementDraft("text_block", "PIT", B)


class TestOcrTolerantReclassification:
    """Real bug found via a live Pair 2 delta run (2026-07-25): tesseract's
    common single-character confusions (0/O, 1/I/l, 5/S, 8/B, 2/Z, 6/G)
    broke these exact-anchored patterns just often enough to reclassify a
    real line_number/valve/tag as a generic text_block, which then made the
    delta engine's same-type matching gate refuse to rescue it against its
    correctly-classified counterpart even at near-zero bbox distance. Only
    ever enabled by the scanned-PDF adapter (ocr_tolerant defaults False).
    """

    def test_disabled_by_default(self):
        # The exact real corrupted string observed live -- without
        # ocr_tolerant, it must stay a plain text_block (native PDF/DWG
        # behavior, unchanged).
        lines = [L('1"-Al-63-9006-AS20-00')]
        assert classify_block_lines(lines) == [
            ElementDraft("text_block", '1"-Al-63-9006-AS20-00', B)
        ]

    def test_line_number_with_lowercase_l_for_capital_i(self):
        # Real example: "AI" (a valid 2-letter service code) OCR'd as "Al"
        # (lowercase L instead of capital I).
        lines = [L('1"-Al-63-9006-AS20-00')]
        result = classify_block_lines(lines, ocr_tolerant=True)
        assert result[0].type == "line_number"
        assert result[0].text == '1"-Al-63-9006-AS20-00'  # original text never rewritten
        assert result[0].attributes["ocr_reclassified"] == "true"

    def test_leading_size_digit_is_not_corrupted_by_a_blind_substitution(self):
        # Regression guard for the exact mistake caught during development:
        # a naive whole-string substitution (e.g. blindly mapping every "1"
        # to "I") would also mangle the line number's own leading size
        # digit ("1" -> "I"), breaking the match. The fix must be
        # position-aware (only the letter-class field is loosened).
        lines = [L('1"-Al-63-9006-AS20-00')]
        result = classify_block_lines(lines, ocr_tolerant=True)
        assert result[0].type == "line_number"

    def test_valve_tag_with_inserted_digit_lookalike(self):
        # Real example from Pair 2: the clean tag "40GT9309" was OCR'd as
        # "40G6T9309" (an inserted "6" inside the letter-class field).
        lines = [L("40G6T9309")]
        result = classify_block_lines(lines, ocr_tolerant=True)
        assert result[0].type == "valve"
        assert result[0].text == "40G6T9309"

    def test_equipment_tag_with_digit_lookalike_letter(self):
        # Plausible OCR confusion: real tag "26-HB-911" misread as
        # "26-H8-911" (8 for B).
        lines = [L("26-H8-911")]
        result = classify_block_lines(lines, ocr_tolerant=True)
        assert result[0].type == "tag"
        assert result[0].text == "26-H8-911"
        assert result[0].attributes["ocr_reclassified"] == "true"

    def test_instrument_loop_function_code_is_normalized_for_vocabulary_check(self):
        # "PIT" OCR'd with a lookalike digit in place of "I" -- the
        # tolerant regex matches structurally, but the captured function
        # code must be normalized back to "PIT" before the
        # INSTRUMENT_FUNCTION_CODES membership check, or it would never
        # pass (a corrupted string can't equal a known vocabulary entry).
        lines = [L("P1T-9062")]
        result = classify_block_lines(lines, ocr_tolerant=True)
        assert result[0].type == "instrument_loop"
        assert result[0].attributes["function"] == "PIT"  # normalized, not "P1T"
        assert result[0].attributes["loop_number"] == "9062"
        assert result[0].text == "P1T-9062"  # displayed text still untouched

    def test_function_code_that_does_not_normalize_to_a_known_code_stays_text_block(self):
        # The tolerant regex matches structurally, but "XYZ" (normalized:
        # still "XYZ") isn't a real instrument function code -- must not be
        # accepted just because it structurally looks tag-shaped.
        lines = [L("XYZ-9062")]
        result = classify_block_lines(lines, ocr_tolerant=True)
        assert result[0] == ElementDraft("text_block", "XYZ-9062", B)

    def test_already_classified_flag_is_not_touched(self):
        # Single-letter flags already carry a "kind" attribute -- the OCR
        # retry must skip anything already specifically classified, not
        # just anything typed text_block.
        lines = [L("U")]
        result = classify_block_lines(lines, ocr_tolerant=True)
        assert result == [ElementDraft("text_block", "U", B, {"kind": "flag"})]

    def test_generic_prose_is_not_falsely_reclassified(self):
        # Safety net: ordinary drawing text must not accidentally acquire
        # tag-shaped structure under the loosened letter-class tolerance.
        words = [
            "VENDOR", "GAS", "COMPRESSOR", "SKID", "DESIGN", "REV", "SHEET",
            "SCALE", "DRAWING", "APPROVED", "PIPING", "MATERIAL", "TYPICAL",
        ]  # fmt: skip
        for word in words:
            result = classify_block_lines([L(word)], ocr_tolerant=True)
            assert result[0].type == "text_block", f"{word!r} was falsely reclassified"
            assert "ocr_reclassified" not in result[0].attributes
