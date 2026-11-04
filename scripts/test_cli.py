"""命令行回归：真实格式 GGUF、量化、多线程和错误参数。"""
import argparse
import copy
import json
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np

from reference_forward import ReferenceModel


def string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def make_gguf(source, raw, path, quant=0):
    c = source["config"]
    metadata = []

    def add(key, kind, value):
        metadata.append(string(key) + struct.pack("<I", kind) + value)

    add("general.architecture", 8, string("qwen2"))
    add("general.alignment", 4, struct.pack("<I", 32))
    for key, field in (("block_count", "n_layers"), ("context_length", "max_seq_len"),
                       ("embedding_length", "dim"), ("feed_forward_length", "hidden_dim"),
                       ("attention.head_count", "n_heads"), ("attention.head_count_kv", "n_kv_heads")):
        add("qwen2." + key, 4, struct.pack("<I", c[field]))
    add("qwen2.rope.freq_base", 6, struct.pack("<f", c["rope_theta"]))
    add("qwen2.attention.layer_norm_rms_epsilon", 6, struct.pack("<f", c["rms_norm_eps"]))
    vocab = [chr(i) for i in range(c["vocab_size"])]
    vocab[:3] = ["<unk>", "<bos>", "<eos>"]
    add("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, len(vocab)) + b"".join(map(string, vocab)))
    add("tokenizer.ggml.model", 8, string("gpt2"))
    add("tokenizer.ggml.pre", 8, string("qwen2"))
    for name, value in (("bos_token_id", 1), ("eos_token_id", 2), ("unknown_token_id", 0)):
        add("tokenizer.ggml." + name, 4, struct.pack("<I", value))
    add("tokenizer.ggml.add_bos_token", 7, b"\x01")
    add("tokenizer.chat_template", 8, string("{% for message in messages %}{{ message['role'] + '\\n' + message['content'] + '\\n' }}{% endfor %}{{ 'assistant\\n' }}"))
    op_names = {"attention_norm": "attn_norm.weight", "wq": "attn_q.weight", "wk": "attn_k.weight",
                "wv": "attn_v.weight", "bq": "attn_q.bias", "bk": "attn_k.bias", "bv": "attn_v.bias",
                "wo": "attn_output.weight", "ffn_norm": "ffn_norm.weight", "w_gate": "ffn_gate.weight",
                "w_up": "ffn_up.weight", "w_down": "ffn_down.weight"}
    top_names = {"token_embedding": "token_embd.weight", "final_norm": "output_norm.weight", "lm_head": "output.weight"}
    data, infos = bytearray(), []
    for item in source["tensors"]:
        name = item["name"]
        if name.startswith("layers."):
            _, layer, op = name.split(".")
            name = f"blk.{layer}.{op_names[op]}"
        else:
            name = top_names[name]
        shape = list(reversed(item["shape"]))
        while len(data) % 32:
            data.append(0)
        payload = raw[item["offset"]:item["offset"] + item["byte_size"]]
        kind = 0
        if quant and len(shape) == 2 and shape[0] % 32 == 0 and name != "token_embd.weight":
            kind = quant
            values = np.frombuffer(payload, dtype="<f4").reshape(-1, 32)
            packed, decoded = [], []
            for block in values:
                if quant == 8:
                    scale = np.float16(np.max(np.abs(block)) / 127)
                    q = np.clip(np.rint(block / float(scale)), -127, 127).astype(np.int8) if scale else np.zeros(32, np.int8)
                    packed.append(scale.tobytes() + q.tobytes())
                    decoded.append(q.astype(np.float32) * np.float32(scale))
                else:
                    offset = np.float16(np.min(block)) if quant == 3 else np.float16(0)
                    scale = np.float16((np.max(block) - np.min(block)) / 15) if quant == 3 else np.float16(block[np.argmax(np.abs(block))] / -8)
                    q = np.rint((block - np.float32(offset)) / np.float32(scale)) if scale else np.zeros(32)
                    q = np.clip(q + (8 if quant == 2 else 0), 0, 15).astype(np.uint8)
                    packed.append(scale.tobytes() + (offset.tobytes() if quant == 3 else b"") + (q[:16] | q[16:] << 4).tobytes())
                    decoded.append((q.astype(np.float32) - (8 if quant == 2 else 0)) * np.float32(scale) + np.float32(offset))
            payload = b"".join(packed)
            raw[item["offset"]:item["offset"] + item["byte_size"]] = np.concatenate(decoded).astype("<f4").tobytes()
        infos.append(string(name) + struct.pack("<I", len(shape)) +
                     struct.pack("<" + "Q" * len(shape), *shape) + struct.pack("<IQ", kind, len(data)))
        data.extend(payload)
    header = b"GGUF" + struct.pack("<IQQ", 3, len(infos), len(metadata)) + b"".join(metadata) + b"".join(infos)
    header += b"\0" * (-len(header) % 32)
    path.write_bytes(header + data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    args = parser.parse_args()
    executable = str(Path(args.exe).resolve())

    def run(arguments, expected=0, input=None):
        result = subprocess.run([executable, *arguments], capture_output=True, text=True, encoding="utf-8", input=input)
        if (result.returncode == 0) != (expected == 0):
            raise AssertionError(f"{arguments}: exit={result.returncode}\n{result.stdout}\n{result.stderr}")
        return result

    with tempfile.TemporaryDirectory(prefix="mini_ai_cli_") as temporary:
        root = Path(temporary)
        source = copy.deepcopy(json.loads(Path("models/tiny/model.json").read_text()))
        source["config"]["rope_type"] = "neox"
        source["tokenizer"] = {"type": "ascii"}
        raw = bytearray(Path("models/tiny/model.bin").read_bytes())
        for layer in range(source["config"]["n_layers"]):
            for name in ("bq", "bk", "bv"):
                size = source["config"]["dim"]
                values = np.linspace(-0.1, 0.15, size, dtype="<f4").tobytes()
                source["tensors"].append({"name": f"layers.{layer}.{name}", "shape": [size], "dtype": "float32",
                                           "offset": len(raw), "byte_size": len(values)})
                raw.extend(values)
        config, weights, gguf = root / "model.json", root / "model.bin", root / "qwen.gguf"
        config.write_text(json.dumps(source))
        weights.write_bytes(raw)
        make_gguf(source, raw, gguf)
        dump = root / "gguf_logits"
        run(["generate", "--model", str(gguf), "-p", "hello", "-n", "4", "--dump-logits", str(dump)])
        tokens = json.loads((dump / "tokens.json").read_text())
        assert tokens["prompt"] == [1, 104, 101, 108, 108, 111]
        generated = []
        for step, (expected, token) in enumerate(ReferenceModel(config, weights).generate(tokens["prompt"], 4)):
            actual = np.fromfile(dump / f"logits_{step:04d}.bin", dtype="<f4")
            assert np.max(np.abs(actual - expected)) < 1e-4
            generated.append(token)
        assert generated == tokens["generated"]
        for kind in (8, 2, 3):
            quantized_raw = bytearray(raw)
            packed_model = root / f"quantized_{kind}.gguf"
            make_gguf(source, quantized_raw, packed_model, kind)
            reference_weights = root / f"decoded_{kind}.bin"
            reference_weights.write_bytes(quantized_raw)
            directory = root / f"gguf_quant_{kind}"
            run(["generate", "--model", str(packed_model), "-p", "hello", "-n", "4", "--dump-logits", str(directory)])
            actual_tokens = json.loads((directory / "tokens.json").read_text())
            generated = []
            for step, (expected, token) in enumerate(ReferenceModel(config, reference_weights).generate(tokens["prompt"], 4)):
                actual = np.fromfile(directory / f"logits_{step:04d}.bin", dtype="<f4")
                assert np.max(np.abs(actual - expected)) < 1e-4
                generated.append(token)
            assert generated == actual_tokens["generated"]
        for quant in ("q8_0", "q4_0", "q4_1"):
            dumps = []
            for threads in (1, 3):
                directory = root / f"{quant}_{threads}"
                result = run(["bench", "models/tiny", "-n", "4", "--quant", quant, "--threads", str(threads),
                              "--dump-logits", str(directory)])
                assert "prefill_tokens_per_second=" in result.stdout
                dumps.append(directory)
            assert (dumps[0] / "tokens.json").read_bytes() == (dumps[1] / "tokens.json").read_bytes()
            for file in dumps[0].glob("logits_*.bin"):
                np.testing.assert_allclose(np.fromfile(file, dtype="<f4"), np.fromfile(dumps[1] / file.name, dtype="<f4"), atol=1e-6)
        run(["run", str(gguf), "-n", "2"], input="hi\n/clear\n/exit\n")
        for arguments in (["generate", "--threads", "-1"], ["generate", "--model", "missing.gguf"],
                          ["generate", "--seed", "-1"], ["generate", "-n", "999999999999999999999"],
                          ["generate", "--backend", "unknown"], ["generate", "--quant", "bad"],
                          ["generate", "-n", "0"], ["generate", "-p", "x" * 128]):
            run(arguments, expected=1)
    print("GGUF, quantization, chat and CLI regression passed")


if __name__ == "__main__":
    main()
