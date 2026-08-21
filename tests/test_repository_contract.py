import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_portfolio_notebook_is_executed() -> None:
    notebook = json.loads(
        (ROOT / "annomi_counselling_dialogue_analysis.ipynb").read_text(encoding="utf-8")
    )
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )


def test_full_pipeline_is_code_only_and_output_free() -> None:
    notebook = json.loads(
        (ROOT / "experiments" / "pipeline_source.ipynb").read_text(encoding="utf-8")
    )
    assert notebook["cells"]
    assert all(cell["cell_type"] == "code" for cell in notebook["cells"])
    assert all(cell["execution_count"] is None for cell in notebook["cells"])
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])
