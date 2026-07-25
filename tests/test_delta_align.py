"""Fast, deterministic alignment unit tests on hand-built minimal canonical
elements (plan §4) — no PDFs, no network, no real embedding model.

Named edge cases from the plan's own framing of "where does alignment
break?" (§14): a moved-and-renamed tag, a note split into two, a duplicate
tag number. Each gets an explicit test below.
"""

import numpy as np
import pytest

from src.config import settings
from src.delta.align import (
    element_key,
    embedding_proximity_match,
    exact_key_match,
    geometry_match,
)
from tests.conftest import FakeEmbedder
from tests.conftest import make_positioned_element as el


class TestElementKey:
    @pytest.mark.parametrize("etype", ["tag", "instrument_loop", "valve", "line_number"])
    def test_keyed_types_use_text(self, etype):
        e = el("26BL9072", element_type=etype)
        assert element_key(e) == "26BL9072"

    def test_note_definition_keyed_by_number(self):
        e = el("30. SOME NOTE", element_type="note", kind="definition", note_number="30")
        assert element_key(e) == "note-def:30"

    def test_note_reference_has_no_key(self):
        e = el("NOTE 30", element_type="note", kind="reference", refs="30")
        assert element_key(e) is None

    @pytest.mark.parametrize("etype", ["setpoint", "dimension", "text_block", "geometry"])
    def test_value_bearing_types_have_no_key(self, etype):
        """Setpoint/dimension are deliberately excluded: exact-keying the
        full text (which includes the value) would never match an edit."""
        e = el("SP = 257 bar (g)", element_type=etype)
        assert element_key(e) is None

    def test_blank_text_has_no_key(self):
        e = el("   ", element_type="tag")
        assert element_key(e) is None


class TestExactKeyMatch:
    def test_matches_identical_keys(self):
        a = el("26BL9072", seq=1)
        b = el("26BL9072", seq=1)
        matched, ua, ub = exact_key_match([a], [b])
        assert len(matched) == 1
        assert matched[0].a.id == a.id and matched[0].b.id == b.id
        assert matched[0].tier == "exact_key"
        assert ua == [] and ub == []

    def test_unmatched_keys_pass_through(self):
        a = el("26BL9072", seq=1)
        b = el("26BL9099", seq=1)
        matched, ua, ub = exact_key_match([a], [b])
        assert matched == []
        assert ua == [a] and ub == [b]

    def test_untyped_elements_never_match_via_tier1(self):
        a = el("SP = 257 bar (g)", element_type="setpoint", seq=1)
        b = el("SP = 260 bar (g)", element_type="setpoint", seq=1)
        matched, ua, ub = exact_key_match([a], [b])
        assert matched == []
        assert ua == [a] and ub == [b]

    def test_duplicate_tag_number_pairs_by_nearest_position(self):
        """Two elements sharing the same key on each side (a data-entry
        duplicate, or two instances of a note-reference callout) must pair
        by nearest bbox, deterministically, each element consumed once."""
        a_near = el("26BL9072", seq=1, x0=0.10, y0=0.10, x1=0.12, y1=0.12)
        a_far = el("26BL9072", seq=2, x0=0.80, y0=0.80, x1=0.82, y1=0.82)
        b_near = el("26BL9072", seq=1, x0=0.11, y0=0.11, x1=0.13, y1=0.13)  # close to a_near
        b_far = el("26BL9072", seq=2, x0=0.79, y0=0.79, x1=0.81, y1=0.81)  # close to a_far

        matched, ua, ub = exact_key_match([a_near, a_far], [b_near, b_far])
        assert len(matched) == 2
        pairs = {(m.a.id, m.b.id) for m in matched}
        assert (a_near.id, b_near.id) in pairs
        assert (a_far.id, b_far.id) in pairs
        assert ua == [] and ub == []

    def test_duplicate_key_count_mismatch_leaves_leftover_unmatched(self):
        a1 = el("NOTE 30", seq=1, x0=0.1, y0=0.1, x1=0.11, y1=0.11)
        a2 = el("NOTE 30", seq=2, x0=0.5, y0=0.5, x1=0.51, y1=0.51)
        a3 = el("NOTE 30", seq=3, x0=0.9, y0=0.9, x1=0.91, y1=0.91)
        b1 = el("NOTE 30", seq=1, x0=0.1, y0=0.1, x1=0.11, y1=0.11)
        b2 = el("NOTE 30", seq=2, x0=0.5, y0=0.5, x1=0.51, y1=0.51)
        matched, ua, ub = exact_key_match([a1, a2, a3], [b1, b2])
        assert len(matched) == 2
        assert ua == [a3]
        assert ub == []


class TestGeometryMatch:
    def test_matches_same_layer_entity_block_within_tolerance(self):
        a = el("", element_type="geometry", seq=1, x0=0.40, y0=0.30, x1=0.42, y1=0.32,
               layer="PIPING", entity_type="INSERT", block_name="VALVE_GATE")
        b = el("", element_type="geometry", seq=1, x0=0.45, y0=0.35, x1=0.47, y1=0.37,
               layer="PIPING", entity_type="INSERT", block_name="VALVE_GATE")
        matched, ua, ub = geometry_match([a], [b])
        assert len(matched) == 1
        assert matched[0].tier == "geometry"
        assert matched[0].bbox_distance == pytest.approx(0.0707, abs=1e-3)

    def test_different_layer_does_not_match(self):
        a = el("", element_type="geometry", seq=1, layer="PIPING", entity_type="LINE")
        b = el("", element_type="geometry", seq=1, layer="INSTRUMENTS", entity_type="LINE")
        matched, ua, ub = geometry_match([a], [b])
        assert matched == []
        assert ua == [a] and ub == [b]

    def test_too_far_does_not_match(self, monkeypatch):
        monkeypatch.setattr(settings, "geometry_match_max_bbox_distance", 0.1)
        a = el("", element_type="geometry", seq=1, x0=0.0, y0=0.0, x1=0.01, y1=0.01,
               layer="PIPING", entity_type="INSERT", block_name="VALVE_GATE")
        b = el("", element_type="geometry", seq=1, x0=0.9, y0=0.9, x1=0.91, y1=0.91,
               layer="PIPING", entity_type="INSERT", block_name="VALVE_GATE")
        matched, ua, ub = geometry_match([a], [b])
        assert matched == []
        assert ua == [a] and ub == [b]

    def test_non_geometry_elements_pass_through_untouched(self):
        tag = el("26BL9072", element_type="tag", seq=1)
        matched, ua, ub = geometry_match([tag], [])
        assert matched == []
        assert ua == [tag]

    def test_removed_geometry_has_no_partner(self):
        """Pair 3's GT3-DEL-DRAIN case: a LINE with no text key and no
        counterpart on the B side must fall through as unmatched, not error."""
        drain = el("", element_type="geometry", seq=1, layer="PIPING", entity_type="LINE")
        matched, ua, ub = geometry_match([drain], [])
        assert matched == []
        assert ua == [drain] and ub == []

    def test_unnamed_primitives_use_a_much_tighter_tolerance(self, monkeypatch):
        """Bare LINE/CIRCLE/... entities (no block_name) have no identity
        beyond position -- they must NOT get the generous named-block
        tolerance, or an unrelated pair of same-layer lines that happen to
        be moderately apart get force-matched to each other."""
        monkeypatch.setattr(settings, "geometry_match_max_bbox_distance", 0.3)
        monkeypatch.setattr(settings, "geometry_match_unnamed_max_bbox_distance", 0.03)
        a = el("", element_type="geometry", x0=0.10, y0=0.10, x1=0.11, y1=0.11,
               layer="PIPING", entity_type="LINE")
        # dist ~0.10: inside the named-block tolerance, outside the unnamed one
        b = el("", element_type="geometry", x0=0.20, y0=0.10, x1=0.21, y1=0.11,
               layer="PIPING", entity_type="LINE")
        matched, ua, ub = geometry_match([a], [b])
        assert matched == []
        assert ua == [a] and ub == [b]

    def test_regression_removed_line_does_not_steal_an_unrelated_added_line(self, monkeypatch):
        """The exact bug found on real Pair 3 data: a removed drain stub
        (measured distance 0.16 from an unrelated added tie-in line) must
        stay unmatched rather than being force-paired with it, while an
        unchanged sibling line on the same layer still matches itself."""
        monkeypatch.setattr(settings, "geometry_match_unnamed_max_bbox_distance", 0.03)
        unchanged_a = el("", element_type="geometry", x0=0.10, y0=0.40, x1=0.60, y1=0.41,
                          layer="PIPING", entity_type="LINE")
        unchanged_b = el("", element_type="geometry", x0=0.10, y0=0.40, x1=0.60, y1=0.41,
                          layer="PIPING", entity_type="LINE")
        removed_drain = el("", element_type="geometry", x0=0.43, y0=0.28, x1=0.44, y1=0.40,
                            layer="PIPING", entity_type="LINE")
        added_tie_in = el("", element_type="geometry", x0=0.59, y0=0.23, x1=0.60, y1=0.40,
                           layer="PIPING", entity_type="LINE")

        matched, ua, ub = geometry_match([unchanged_a, removed_drain], [unchanged_b, added_tie_in])
        assert len(matched) == 1
        assert matched[0].a.id == unchanged_a.id and matched[0].b.id == unchanged_b.id
        assert ua == [removed_drain]
        assert ub == [added_tie_in]


class TestEmbeddingProximityMatch:
    def test_tight_proximity_overrides_low_similarity(self):
        """The calibrated real case: '19057' -> '20500' at (nearly) the same
        slot must match even though embedding similarity is low (~0.10)."""

        a = el("19057", element_type="text_block", seq=1, x0=0.100, y0=0.600, x1=0.150, y1=0.610)
        b = el("20500", element_type="text_block", seq=1, x0=0.101, y0=0.601, x1=0.140, y1=0.611)
        embedder = FakeEmbedder({"19057": [1.0, 0.0], "20500": [0.1, 0.995]})  # sim ~0.1

        matched, ua, ub = embedding_proximity_match([a], [b], embedder)
        assert len(matched) == 1
        assert matched[0].match_score >= 0.9  # position-decisive override band
        assert ua == [] and ub == []

    def test_loose_proximity_requires_high_similarity(self):
        a = el("SP = 257 bar (g)", element_type="setpoint", x0=0.10, y0=0.10, x1=0.15, y1=0.11)
        b = el("SP = 260 bar (g)", element_type="setpoint", x0=0.16, y0=0.10, x1=0.21, y1=0.11)
        embedder = FakeEmbedder(
            {"SP = 257 bar (g)": [1.0, 0.0], "SP = 260 bar (g)": [0.95, np.sqrt(1 - 0.95**2)]}
        )
        matched, ua, ub = embedding_proximity_match([a], [b], embedder)
        assert len(matched) == 1
        # loose-band score is similarity scaled down (<0.9) so it never
        # outranks a tight-band position-decisive match; see align.py.
        assert matched[0].match_score == pytest.approx(0.95 * 0.89)
        assert matched[0].match_score < 0.9

    def test_loose_proximity_with_low_similarity_rejected(self):
        """The honest break case: text changed enough that embedding
        similarity falls below threshold, and the pair is too far apart for
        position alone to rescue it -- correctly reported as no match."""

        a = el("26BL9072", element_type="valve", seq=1, x0=0.10, y0=0.10, x1=0.12, y1=0.11)
        b = el("43GT1005", element_type="valve", seq=1, x0=0.10, y0=0.20, x1=0.12, y1=0.21)
        embedder = FakeEmbedder({"26BL9072": [1.0, 0.0], "43GT1005": [0.3, np.sqrt(1 - 0.09)]})
        matched, ua, ub = embedding_proximity_match([a], [b], embedder)
        assert matched == []
        assert ua == [a] and ub == [b]

    def test_too_far_rejected_regardless_of_similarity(self):
        a = el("FLOW RATE NOTE", element_type="text_block", x0=0.05, y0=0.05, x1=0.10, y1=0.06)
        b = el("FLOW RATE NOTE", element_type="text_block", x0=0.90, y0=0.90, x1=0.95, y1=0.91)
        # identical text -> identical vector, but too far apart -- distinct, unrelated occurrences
        embedder = FakeEmbedder({"FLOW RATE NOTE": [1.0, 0.0]})
        matched, ua, ub = embedding_proximity_match([a], [b], embedder)
        assert matched == []

    def test_short_text_below_min_length_is_never_embedded(self):

        a = el("*", element_type="text_block", seq=1)
        b = el("*", element_type="text_block", seq=1)
        embedder = FakeEmbedder({})  # would KeyError if embed() were ever called
        matched, ua, ub = embedding_proximity_match([a], [b], embedder)
        assert matched == []
        assert ua == [a] and ub == [b]

    def test_global_greedy_prefers_exact_self_match_over_ambiguous_candidate(self):
        """Calibration case: an unchanged element ('776 NOTE 28') sitting
        near a genuinely-edited value ('19057' -> '20500') must claim its
        own perfect match first, so it can't be mistakenly stolen as the
        edited value's partner."""

        flow_a = el("19057", element_type="text_block", x0=0.10, y0=0.600, x1=0.15, y1=0.610)
        flow_b = el("20500", element_type="text_block", x0=0.10, y0=0.601, x1=0.14, y1=0.611)
        duty_a = el("776 NOTE 28", element_type="text_block", x0=0.10, y0=0.590, x1=0.20, y1=0.600)
        duty_b = el("776 NOTE 28", element_type="text_block", x0=0.10, y0=0.590, x1=0.20, y1=0.600)

        embedder = FakeEmbedder(
            {
                "19057": [1.0, 0.0, 0.0],
                "20500": [0.1, 0.995, 0.0],  # low sim to 19057
                "776 NOTE 28": [0.0, 0.0, 1.0],  # identical text on both sides -> identical vector
            }
        )
        matched, ua, ub = embedding_proximity_match(
            [flow_a, duty_a], [flow_b, duty_b], embedder
        )
        pairs = {(m.a.id, m.b.id): m.tier for m in matched}
        assert (flow_a.id, flow_b.id) in pairs
        assert (duty_a.id, duty_b.id) in pairs
        assert len(matched) == 2


class TestPageScoping:
    """Real bug found via Pair 5 (multi-page stress set, plan §6/§12 Phase
    12): every sample pair through Pair 4 was single-page, so nothing had
    ever exercised two elements sharing identical *normalized* (page-
    relative) bbox coordinates while sitting on different pages. All three
    tiers used only normalized bbox distance, which is blind to page, so a
    same-text/same-geometry element repeated per-page could be matched
    across pages -- confirmed with a 2-page repro where a genuinely
    unmatched page-1 tag was reported "removed" while the actually-removed
    page-0 tag was silently treated as matched. Fixed by scoping every
    tier's candidate generation to same-page pairs only.
    """

    def test_tier1_does_not_cross_pages(self):
        # Same text on both pages of A; B keeps only the page-1 copy --
        # the correct read is "page 0's tag was removed", not the reverse.
        a_page0 = el("DUPTAG", seq=1, page=0)
        a_page1 = el("DUPTAG", seq=2, page=1)
        b_page1 = el("DUPTAG", seq=1, page=1)

        matched, ua, ub = exact_key_match([a_page0, a_page1], [b_page1])
        assert len(matched) == 1
        assert matched[0].a.id == a_page1.id  # the real, same-page match
        assert matched[0].b.id == b_page1.id
        assert ua == [a_page0]  # correctly reported as the removed one
        assert ub == []

    def test_tier1_matches_normally_within_a_non_zero_page(self):
        a = el("26BL9072", seq=1, page=3)
        b = el("26BL9072", seq=1, page=3)
        matched, _, _ = exact_key_match([a], [b])
        assert len(matched) == 1

    def test_tier2_does_not_cross_pages(self):
        kwargs = {
            "element_type": "geometry", "layer": "PIPING",
            "entity_type": "INSERT", "block_name": "VALVE_GATE",
        }  # fmt: skip
        a_page0 = el("", x0=0.40, y0=0.30, x1=0.42, y1=0.32, page=0, **kwargs)
        a_page1 = el("", x0=0.40, y0=0.30, x1=0.42, y1=0.32, page=1, **kwargs)
        b_page1 = el("", x0=0.41, y0=0.31, x1=0.43, y1=0.33, page=1, **kwargs)

        matched, ua, ub = geometry_match([a_page0, a_page1], [b_page1])
        assert len(matched) == 1
        assert matched[0].a.id == a_page1.id
        assert ua == [a_page0]
        assert ub == []

    def test_tier3_does_not_cross_pages(self):
        kwargs = {"element_type": "text_block", "x0": 0.10, "y0": 0.60, "x1": 0.15, "y1": 0.61}
        a_page0 = el("FLOW RATE NOTE", page=0, **kwargs)
        a_page1 = el("FLOW RATE NOTE", page=1, **kwargs)
        b_page1 = el("FLOW RATE NOTE", page=1, **kwargs)
        embedder = FakeEmbedder({"FLOW RATE NOTE": [1.0, 0.0]})

        matched, ua, ub = embedding_proximity_match([a_page0, a_page1], [b_page1], embedder)
        assert len(matched) == 1
        assert matched[0].a.id == a_page1.id
        assert ua == [a_page0]
        assert ub == []


class TestMovedAndRenamedTag:
    """Plan-named edge case: a tag that is both moved and renamed breaks
    tier 1 (the key changed) by construction. What happens next depends on
    whether the new name is still recognizably related -- both outcomes are
    legitimate and are pinned down here.
    """

    def test_recovers_via_tier3_when_name_change_is_small(self):

        old = el("26BL9072", element_type="valve", x0=0.40, y0=0.40, x1=0.42, y1=0.41)
        # moved + renamed
        new = el("26BL9099", element_type="valve", x0=0.44, y0=0.44, x1=0.46, y1=0.45)
        embedder = FakeEmbedder({"26BL9072": [1.0, 0.0], "26BL9099": [0.93, np.sqrt(1 - 0.93**2)]})

        matched, ua, ub = embedding_proximity_match([old], [new], embedder)
        assert len(matched) == 1  # recovered despite tier 1 failing on the key change

    def test_does_not_match_when_name_and_position_both_diverge(self):

        old = el("26BL9072", element_type="valve", seq=1, x0=0.10, y0=0.10, x1=0.12, y1=0.11)
        new = el("43GT1005", element_type="valve", seq=1, x0=0.70, y0=0.70, x1=0.72, y1=0.71)
        embedder = FakeEmbedder({"26BL9072": [1.0, 0.0], "43GT1005": [0.3, np.sqrt(1 - 0.09)]})

        matched, ua, ub = embedding_proximity_match([old], [new], embedder)
        assert matched == []  # honest break: reported as separate removed + added


class TestNoteSplitIntoTwo:
    def test_split_note_resolves_to_one_modified_plus_one_added(self):
        """A note split into two in the B revision: the first half keeps the
        original note_number (tier 1 matches it as modified); the second
        half is a genuinely new note with no A-side counterpart left to
        claim, since the original is already consumed -- it must end up
        unmatched (added), not incorrectly stolen from the first match.
        """

        original = el(
            "5. OIL CHANGE BY USING TEMPORARY ARRANGEMENT WITH HOSES.",
            element_type="note", x0=0.10, y0=0.70, x1=0.40, y1=0.71,
            kind="definition", note_number="5",
        )
        split_a = el(
            "5. OIL CHANGE.", element_type="note", x0=0.10, y0=0.70, x1=0.20, y1=0.71,
            kind="definition", note_number="5",
        )
        split_b = el(
            "5a. USE TEMPORARY HOSES.", element_type="note", x0=0.10, y0=0.72, x1=0.25, y1=0.73,
            kind="definition", note_number="5a",
        )

        tier1_matched, ua, ub = exact_key_match([original], [split_a, split_b])
        assert len(tier1_matched) == 1
        assert tier1_matched[0].a.id == original.id
        assert tier1_matched[0].b.id == split_a.id
        assert ua == []
        assert ub == [split_b]

        # split_b has no remaining A-side candidate (original is consumed) -> stays unmatched
        embedder = FakeEmbedder({})
        matched3, ua3, ub3 = embedding_proximity_match(ua, ub, embedder)
        assert matched3 == []
        assert ub3 == [split_b]
