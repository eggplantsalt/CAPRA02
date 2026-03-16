"""I/O helpers and schema for CAPRA mined supervision records."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass
class SupervisionRecord:
    """One mined CAPRA supervision record."""

    observation: Dict[str, Any]
    instruction: str
    base_action: List[Any]
    safer_action: List[Any]
    weight: float
    candidate_stats: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def write_supervision_jsonl(records: Iterable[SupervisionRecord], output_path: str) -> int:
    """Write supervision records to JSONL and return number of lines written."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=True) + "\n")
            count += 1
    return count


def read_supervision_jsonl(input_path: str) -> List[Dict[str, Any]]:
    """Read previously written JSONL supervision file."""
    path = Path(input_path)
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows
