"""比较两份 logits 和 greedy token 序列。"""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(cpp_dir, py_dir, tolerance=1e-4):
    cpp_dir, py_dir = Path(cpp_dir), Path(py_dir)
    a, b = sorted(cpp_dir.glob("logits_*.bin")), sorted(py_dir.glob("logits_*.bin"))
    if not a or [p.name for p in a] != [p.name for p in b]:
        raise AssertionError("logit step lists differ or are empty")
    worst = 0.0
    for cpp, reference in zip(a, b):
        actual, expected = np.fromfile(cpp, dtype="<f4"), np.fromfile(reference, dtype="<f4")
        if actual.shape != expected.shape or actual.size == 0:
            raise AssertionError(f"shape mismatch: {cpp.name}")
        if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
            raise AssertionError(f"non-finite logits: {cpp.name}")
        error = float(np.max(np.abs(actual - expected)))
        worst = max(worst, error)
        if error > tolerance:
            raise AssertionError(f"{cpp.name}: max error {error} > {tolerance}")
    if json.loads((cpp_dir / "tokens.json").read_text()) != json.loads((py_dir / "tokens.json").read_text()):
        raise AssertionError("token sequences differ")
    return worst


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp-dir", required=True)
    parser.add_argument("--py-dir", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    print(f"max_abs_error={compare(args.cpp_dir, args.py_dir, args.tolerance):.8g}")
