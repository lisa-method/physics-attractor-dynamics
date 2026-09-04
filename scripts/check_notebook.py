"""Check notebook syntax and keep generated output out of the repository.

Use --clean after an interactive run. The complete original is backed up to
the Git-ignored output/ directory before outputs and execution metadata change.
This does not execute experiments or validate their scientific conclusions.
"""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/attractor_todo.ipynb"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    original = NOTEBOOK.read_bytes()
    notebook = json.loads(original)
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    for index, cell in enumerate(code_cells, start=1):
        compile("".join(cell["source"]), f"notebook code cell {index}", "exec")
    for directory in ("notebooks", "scripts", "tests"):
        for path in (ROOT / directory).glob("*.py"):
            compile(path.read_text(encoding="utf-8"), str(path.relative_to(ROOT)), "exec")

    dirty = any(
        c.get("outputs") or c.get("execution_count") is not None
        or "execution" in c.get("metadata", {})
        for c in code_cells
    ) or "widgets" in notebook.get("metadata", {})
    if dirty and args.clean:
        digest = hashlib.sha256(original).hexdigest()[:16]
        backup = ROOT / "output" / f"attractor_todo.executed.{digest}.ipynb"
        backup.parent.mkdir(exist_ok=True)
        if backup.exists():
            if backup.read_bytes() != original:
                raise RuntimeError("Backup name collision; notebook was not modified")
        else:
            backup.write_bytes(original)
        for cell in code_cells:
            cell["outputs"] = []
            cell["execution_count"] = None
            cell.get("metadata", {}).pop("execution", None)
        notebook.get("metadata", {}).pop("widgets", None)
        NOTEBOOK.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print(f"Full local backup: {backup.relative_to(ROOT)}")
    elif dirty:
        raise SystemExit(
            "Notebook contains outputs or execution metadata. Run "
            "uv run --no-sync python scripts/check_notebook.py --clean"
        )
    print(f"PASS: {len(code_cells)} notebook code cells and Python sources compile; outputs are clear")


if __name__ == "__main__":
    main()
