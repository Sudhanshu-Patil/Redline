from src.ingest.pdf_native import ElementDraft, classify_block_lines

B = (0.0, 0.0, 10.0, 10.0)  # placeholder bbox; classify_block_lines doesn't inspect its values


def L(text: str, bbox=B):
    return (text, bbox)


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
