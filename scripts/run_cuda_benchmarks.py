"""在 CUDA 机器上比较 CPU/GPU logits，并记录运行统计。"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from compare_logits import compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--model", default="models/tiny")
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("-n", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default="cuda_benchmarks.json")
    args = parser.parse_args()
    executable = str(Path(args.exe).resolve())
    info = subprocess.run([executable, "--cuda-info"], capture_output=True, text=True, check=True)
    if "unavailable" in info.stdout:
        parser.error("the executable has no CUDA backend")
    results = {"devices": info.stdout, "runs": []}
    with tempfile.TemporaryDirectory(prefix="mini_ai_cuda_") as temporary:
        for quant in ("", "q8_0", "q4_0", "q4_1"):
            outputs = {}
            for backend in ("cpu", "cuda"):
                directory = Path(temporary) / (quant or "original") / backend
                command = [executable, "bench", args.model, "-p", args.prompt, "-n", str(args.n),
                           "--backend", backend, "--device", str(args.device), "--seed", "1",
                           "--dump-logits", str(directory)]
                if quant:
                    command += ["--quant", quant]
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                outputs[backend] = {"directory": directory, "log": result.stdout}
            error = compare(outputs["cuda"]["directory"], outputs["cpu"]["directory"])
            results["runs"].append({"quant": quant or "original", "max_abs_error": error,
                                    "cpu": outputs["cpu"]["log"], "cuda": outputs["cuda"]["log"]})
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"CUDA comparisons passed; report={args.output}")


if __name__ == "__main__":
    main()
