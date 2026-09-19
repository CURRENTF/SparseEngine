"""Revalidate matched pre/post full-model windows with the shared summarizer."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--baseline-root", type=Path, required=True)
p.add_argument("--new-root", type=Path, required=True)
p.add_argument("--output", type=Path)
args = p.parse_args()
cases = []
for source, engine in ((args.baseline_root, "Legacy Triton"), (args.new_root, "SM Triton")):
    for name, concurrency in (("04_glm_common8", 8), ("06_glm_wave64", 64)):
        cases.append({"model": "GLM-4.7-Flash", "method": "H2O", "engine": engine,
                      "directory": str((source / "queue" / name).resolve()), "concurrency": concurrency, "phase": "measure"})
case_path = args.new_root / "comparison_cases.json"
case_path.write_text(json.dumps({"cases": cases}, indent=2) + "\n")
summarizer = Path(__file__).resolve().parents[1] / "sparseengine_vs_vortex/summarize.py"
output = args.output or args.new_root / "comparison.json"
subprocess.run([sys.executable, str(summarizer), "--root", str(args.new_root), "--cases", str(case_path),
                "--output", str(output)], check=True)
# The shared validator also serves Vortex comparisons. This run compares the
# same native H2O implementation, so its H2O-like algorithm caveat does not apply.
result = json.loads(output.read_text())
result["caveats"] = [
    "Same historical H2O implementation and workload; only three MLA schedule/provider files replaced",
    "32K/512 TP1 boundary-synchronized diagnostic, not the 128K/2K per-step-synchronized figure",
    "Three repetitions on H100 only; not a cross-GPU performance guarantee",
    "Wave2 admission excluded at B64; request metrics are boundary-perturbed",
]
output.write_text(json.dumps(result, indent=2) + "\n")
