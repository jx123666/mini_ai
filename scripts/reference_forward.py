"""JSON+BIN 模型的 NumPy 参考实现。"""
import argparse
import json
from pathlib import Path

import numpy as np


class ReferenceModel:
    def __init__(self, config_path, weights_path):
        manifest = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.config = manifest["config"]
        raw = Path(weights_path).read_bytes()
        self.weights = {
            item["name"]: np.frombuffer(raw, dtype="<f4", count=item["byte_size"] // 4,
                                       offset=item["offset"]).reshape(item["shape"]).copy()
            for item in manifest["tensors"]
        }
        c = self.config
        shape = (c["n_layers"], c["max_seq_len"], c["n_kv_heads"], c["dim"] // c["n_heads"])
        self.keys = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)

    def norm(self, x, weight):
        return x / np.sqrt(np.mean(x * x, dtype=np.float32) + np.float32(self.config["rms_norm_eps"])) * weight

    def rope(self, x, pos):
        x = x.copy()
        dim = x.shape[-1]
        freq = np.float32(pos) / np.power(np.float32(self.config["rope_theta"]),
                                         np.arange(0, dim, 2, dtype=np.float32) / dim)
        a = np.arange(dim // 2) if self.config.get("rope_type") == "neox" else np.arange(0, dim, 2)
        b = a + dim // 2 if self.config.get("rope_type") == "neox" else a + 1
        first, second = x[:, a].copy(), x[:, b].copy()
        x[:, a] = first * np.cos(freq) - second * np.sin(freq)
        x[:, b] = first * np.sin(freq) + second * np.cos(freq)
        return x

    def forward(self, token, pos):
        c, w = self.config, self.weights
        hd = c["dim"] // c["n_heads"]
        x = w["token_embedding"][token].copy()
        for layer in range(c["n_layers"]):
            prefix = f"layers.{layer}."
            h = self.norm(x, w[prefix + "attention_norm"])
            qkv = [w[prefix + "w" + key] @ h + w.get(prefix + "b" + key, np.float32(0))
                   for key in "qkv"]
            q = self.rope(qkv[0].reshape(c["n_heads"], hd), pos)
            self.keys[layer, pos] = self.rope(qkv[1].reshape(c["n_kv_heads"], hd), pos)
            self.values[layer, pos] = qkv[2].reshape(c["n_kv_heads"], hd)
            out = np.empty_like(q)
            for head in range(c["n_heads"]):
                kv = head // (c["n_heads"] // c["n_kv_heads"])
                scores = self.keys[layer, :pos + 1, kv] @ q[head] / np.sqrt(np.float32(hd))
                probs = np.exp(scores - np.max(scores))
                probs /= np.sum(probs, dtype=np.float32)
                out[head] = probs @ self.values[layer, :pos + 1, kv]
            x += w[prefix + "wo"] @ out.reshape(-1)
            h = self.norm(x, w[prefix + "ffn_norm"])
            gate, up = w[prefix + "w_gate"] @ h, w[prefix + "w_up"] @ h
            x += w[prefix + "w_down"] @ (gate / (1 + np.exp(-gate)) * up)
        return w["lm_head"] @ self.norm(x, w["final_norm"])

    def generate(self, tokens, count):
        if not tokens or len(tokens) + count - 1 > self.config["max_seq_len"]:
            raise ValueError("invalid prompt or generation length")
        for pos, token in enumerate(tokens):
            logits = self.forward(token, pos)
        for step in range(count):
            token = int(np.argmax(logits))
            yield logits.copy(), token
            if token == 2 or step + 1 == count:
                break
            pos += 1
            logits = self.forward(token, pos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="models/tiny/model.json")
    parser.add_argument("--weights", default="models/tiny/model.bin")
    parser.add_argument("--tokens", default="1,104,101,108,108,111")
    parser.add_argument("-n", type=int, default=16)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.n <= 0:
        parser.error("-n must be positive")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokens = [int(token) for token in args.tokens.split(",")]
    generated = []
    for step, (logits, token) in enumerate(ReferenceModel(args.config, args.weights).generate(tokens, args.n)):
        logits.astype("<f4").tofile(output / f"logits_{step:04d}.bin")
        generated.append(token)
    (output / "tokens.json").write_text(json.dumps({"prompt": tokens, "generated": generated}), encoding="utf-8")


if __name__ == "__main__":
    main()
