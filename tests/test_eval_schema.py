"""Tests for the QA/HumanLabel schema additions to eval/schema.py."""

import pytest
from pydantic import ValidationError

from eval.schema import HumanLabel, QAItem, load_human_labels, load_qa_dataset


class TestQAItem:
    def test_valid_item(self):
        item = QAItem(
            qa_id="QA-01", pair_id="pair1", question="q?", answerable=True,
            expected_citation_texts=["text"], expected_answer_summary="summary",
        )  # fmt: skip
        assert item.held_out is False

    def test_unanswerable_item_needs_no_citations(self):
        item = QAItem(
            qa_id="QA-01", pair_id="pair1", question="q?", answerable=False,
            expected_answer_summary="not present",
        )  # fmt: skip
        assert item.expected_citation_texts == []


class TestHumanLabel:
    def test_valid_scores(self):
        label = HumanLabel(qa_id="QA-01", correctness=3, groundedness=5)
        assert label.correctness == 3

    def test_score_above_five_rejected(self):
        with pytest.raises(ValidationError):
            HumanLabel(qa_id="QA-01", correctness=6, groundedness=3)

    def test_score_below_one_rejected(self):
        with pytest.raises(ValidationError):
            HumanLabel(qa_id="QA-01", correctness=0, groundedness=3)


class TestLoadQaDataset:
    def test_loads_real_pair1_dataset(self):
        from pathlib import Path

        dataset = load_qa_dataset(Path("eval/datasets/qa_pair1.json"))
        assert dataset.pair_id == "pair1"
        assert len(dataset.items) > 0
        assert all(isinstance(i, QAItem) for i in dataset.items)

    def test_round_trips_through_json(self, tmp_path):
        item = QAItem(
            qa_id="QA-01", pair_id="p", question="q?", answerable=True,
            expected_citation_texts=["a"], expected_answer_summary="s", held_out=True,
        )  # fmt: skip
        from eval.schema import QADataset

        path = tmp_path / "qa.json"
        path.write_text(QADataset(pair_id="p", items=[item]).model_dump_json(), encoding="utf-8")
        loaded = load_qa_dataset(path)
        assert loaded.items[0].qa_id == "QA-01"
        assert loaded.items[0].held_out is True


class TestLoadHumanLabels:
    def test_round_trips_through_json(self, tmp_path):
        from eval.schema import HumanLabelSet

        label_set = HumanLabelSet(
            pair_id="pair1",
            labels=[HumanLabel(qa_id="QA-01", correctness=4, groundedness=5, notes="looks right")],
        )
        path = tmp_path / "labels.json"
        path.write_text(label_set.model_dump_json(), encoding="utf-8")
        loaded = load_human_labels(path)
        assert loaded.labels[0].correctness == 4
        assert loaded.labels[0].notes == "looks right"
