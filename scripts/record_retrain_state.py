from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Record an already validated sweep as the current retrain state.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--pipeline-version", type=int, default=4)
    args = parser.parse_args()

    snapshot = ROOT / "data" / "lmarena_snapshot.json"
    metrics_path = ROOT / "artifacts" / args.run_name / "metrics.json"
    if not snapshot.exists() or not metrics_path.exists():
        raise FileNotFoundError("snapshot and promoted run metrics must exist before recording state")
    state = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": args.pipeline_version,
        "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "run_name": args.run_name,
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
        "source": "validated_h100_sweep",
    }
    output = ROOT / "data" / "continuous_retrain_state.json"
    output.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
