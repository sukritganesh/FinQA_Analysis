"""Small local entrypoints for interacting with the scaffold."""

from __future__ import annotations

from src.data.schemas import FinQAExample, RuntimeInputs
from src.graph.state import PipelineState
from src.graph.workflow import FinQAPipeline
from src.llm.client import PlaceholderModelClient
from src.retrieval.simple import KeywordOverlapRetriever


def run_demo_question(question: str) -> str:
    """Run the starter pipeline on a synthetic local example."""
    example = FinQAExample(
        runtime=RuntimeInputs(
            example_id="demo-example",
            filename=None,
            question=question,
            pre_text=["Revenue increased from $10 to $12 year over year."],
            post_text=["Operating margin improved in the same period."],
            table=[
                ["Metric", "2020", "2021"],
                ["Revenue", "10", "12"],
                ["Operating income", "3", "4"],
            ],
        ),
    )

    pipeline = FinQAPipeline(
        retriever=KeywordOverlapRetriever(),
        model_client=PlaceholderModelClient(),
        top_k=3,
    )
    state = pipeline.run(PipelineState(example=example))
    return (
        f"question={question!r}, "
        f"retrieved={len(state.retrieved_evidence)}, "
        f"answer={state.final_answer!r}, "
        f"errors={state.errors}"
    )
