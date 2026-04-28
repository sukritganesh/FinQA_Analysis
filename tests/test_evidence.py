from __future__ import annotations

from pathlib import Path

from src.data.evidence import build_evidence_units, render_table_row
from src.data.loader import load_finqa_examples
from src.data.schemas import FinQAExample, RuntimeInputs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "test.json"


def test_render_table_row_matches_finqa_gold_style_with_stub_header() -> None:
    rendered = render_table_row(
        ["company", "payments volume ( billions )", "total transactions ( billions )"],
        ["american express", "637", "5.0"],
    )

    assert (
        rendered
        == "company the american express of payments volume ( billions ) is 637 ; "
        "the american express of total transactions ( billions ) is 5.0 ;"
    )


def test_render_table_row_matches_finqa_gold_style_without_stub_header() -> None:
    rendered = render_table_row(
        ["", "amount ( in millions )"],
        ["2014 net revenue", "$ 5735"],
    )

    assert rendered == "the 2014 net revenue of amount ( in millions ) is $ 5735 ;"


def test_render_table_row_normalizes_finqa_punctuation() -> None:
    rendered = render_table_row(
        ["property:", "shares ( thous. )", ""],
        ["total:", "1060", "-1 ( 1 )"],
    )

    assert rendered == "property : the total : of shares ( thous . ) is 1060 ; the total : of is -1 ( 1 ) ;"


def test_build_evidence_units_returns_text_and_table_units() -> None:
    example = FinQAExample(
        runtime=RuntimeInputs(
            example_id="sample",
            filename=None,
            question="What was revenue?",
            pre_text=["Revenue improved. Margins expanded."],
            post_text=["Cash flow stayed positive."],
            table=[
                ["Metric", "2021"],
                ["Revenue", "12"],
            ],
        )
    )

    units = build_evidence_units(example)

    by_id = {unit.evidence_id: unit for unit in units}

    assert by_id["text_0"].source == "text"
    assert by_id["text_0"].text == "Revenue improved. Margins expanded."
    assert by_id["text_0"].metadata["source_section"] == "pre_text"
    assert by_id["text_1"].metadata["source_section"] == "post_text"
    assert by_id["table_1"].source == "table"
    assert by_id["table_1"].text == "Metric the Revenue of 2021 is 12 ;"
    assert by_id["table_1"].metadata["row_name"] == "Revenue"


def test_build_evidence_units_can_match_known_gold_support_text() -> None:
    example = FinQAExample(
        runtime=RuntimeInputs(
            example_id="sample",
            filename=None,
            question="What is the average payment volume?",
            pre_text=[],
            post_text=[],
            table=[
                ["company", "payments volume ( billions )", "total volume ( billions )", "total transactions ( billions )"],
                ["visa inc. ( 1 )", "$ 2457", "$ 3822", "50.3"],
                ["american express", "637", "647", "5.0"],
            ],
        )
    )

    units = {unit.evidence_id: unit for unit in build_evidence_units(example)}

    assert (
        units["table_2"].text
        == "company the american express of payments volume ( billions ) is 637 ; "
        "the american express of total volume ( billions ) is 647 ; "
        "the american express of total transactions ( billions ) is 5.0 ;"
    )


def test_build_evidence_units_matches_real_test_json_table_gold_support() -> None:
    examples = {
        example.runtime.example_id: example
        for example in load_finqa_examples(TEST_DATA_PATH)
    }
    example_ids = [
        "ETR/2016/page_23.pdf-2",
        "INTC/2015/page_41.pdf-4",
        "FIS/2010/page_70.pdf-2",
    ]

    for example_id in example_ids:
        example = examples[example_id]
        units = {unit.evidence_id: unit for unit in build_evidence_units(example)}
        table_support = {
            evidence_id: text
            for evidence_id, text in example.gold.supporting_facts.items()
            if evidence_id.startswith("table_")
        }

        assert table_support
        for evidence_id, gold_text in table_support.items():
            assert units[evidence_id].text == gold_text
