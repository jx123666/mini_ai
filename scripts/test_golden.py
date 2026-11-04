"""运行 C++/NumPy 数值回归，覆盖普通 RoPE 和带偏置的 NeoX RoPE。"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from compare_logits import compare
from reference_forward import ReferenceModel


def check(executable, config, weights, directory, mode):
    cpp, reference = directory / "cpp", directory / "reference"
    subprocess.run([executable, "generate", "--config", str(config), "--model", str(weights),
                    "-p", "hello", "-n", "8", "--dump-logits", str(cpp),
                    "--matmul", mode, "--threads", "3"], check=True, capture_output=True)
    metadata = json.loads((cpp / "tokens.json").read_text())
    reference.mkdir(parents=True)
    generated = []
    for step, (logits, token) in enumerate(ReferenceModel(config, weights).generate(metadata["prompt"], 8)):
        logits.astype("<f4").tofile(reference / f"logits_{step:04d}.bin")
        generated.append(token)
    (reference / "tokens.json").write_text(json.dumps({"prompt": metadata["prompt"], "generated": generated}))
    error = compare(cpp, reference)
    print(f"{config.name} {mode}: max_abs_error={error:.8g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    args = parser.parse_args()
    executable = str(Path(args.exe).resolve())
    original = json.loads(Path("models/tiny/model.json").read_text())
    with tempfile.TemporaryDirectory(prefix="mini_ai_golden_") as temporary:
        root = Path(temporary)
        for variant in ("normal", "neox"):
            folder = root / variant
            folder.mkdir()
            manifest = copy.deepcopy(original)
            manifest["config"]["rope_type"] = variant
            manifest["tokenizer"] = {"type": "ascii"}
            raw = bytearray(Path("models/tiny/model.bin").read_bytes())
            if variant == "neox":
                c = manifest["config"]
                c["n_kv_heads"] = c["n_heads"] // 2
                original_raw, raw = raw, bytearray()
                for item in manifest["tensors"]:
                    values = np.frombuffer(original_raw, dtype="<f4", count=item["byte_size"] // 4,
                                           offset=item["offset"]).reshape(item["shape"])
                    if item["name"].endswith((".wk", ".wv")):
                        values = values[:values.shape[0] // 2]
                    item["shape"] = list(values.shape)
                    item["offset"], item["byte_size"] = len(raw), values.nbytes
                    raw.extend(values.tobytes())
                for layer in range(c["n_layers"]):
                    for name, size in (("bq", c["dim"]), ("bk", c["n_kv_heads"] * c["dim"] // c["n_heads"]),
                                       ("bv", c["n_kv_heads"] * c["dim"] // c["n_heads"])):
                        values = np.linspace(-0.1, 0.2, size, dtype="<f4").tobytes()
                        manifest["tensors"].append({"name": f"layers.{layer}.{name}", "shape": [size],
                                                    "dtype": "float32", "offset": len(raw), "byte_size": len(values)})
                        raw.extend(values)
            config, weights = folder / "model.json", folder / "model.bin"
            config.write_text(json.dumps(manifest), encoding="utf-8")
            weights.write_bytes(raw)
            for mode in ("naive", "threaded", "simd", "threaded-simd"):
                check(executable, config, weights, folder / mode, mode)


if __name__ == "__main__":
    main()
