"""Export the EvalSuite JSON Schema for evalset YAML files (IDE autocomplete/validation).

poetry run python scripts/export_schema.py
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_eval.eval.case import EvalSuite

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    """Write the EvalSuite JSON Schema to evalsets/schema/."""
    schema_dir = ROOT / "evalsets" / "schema"
    schema_dir.mkdir(exist_ok=True)
    schema_path = schema_dir / "eval_suite.schema.json"
    schema_path.write_text(json.dumps(EvalSuite.model_json_schema(), indent=2) + "\n")
    print(f"Wrote {schema_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
