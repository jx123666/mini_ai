#从GGUF metadata导出词表，给BpeTokenizer那三份文件用
#输出：vocab.json / merges.txt / special_tokens.json / 可选chat_template.txt

import argparse
import json
import os
import struct
import sys


def read_gguf_string(f):
    #GGUF字符串：小端uint64长度 + utf-8字节
    slen = struct.unpack("<Q", f.read(8))[0]
    return f.read(slen).decode("utf-8", errors="replace")


def skip_gguf_value(f, vtype):
    #不关心的字段按类型跳过，避免后面key错位
    if vtype in (0, 1, 7):
        f.read(1)
    elif vtype in (2, 3):
        f.read(2)
    elif vtype in (4, 5, 6):
        f.read(4)
    elif vtype == 8:
        read_gguf_string(f)
    elif vtype == 9:
        etype = struct.unpack("<I", f.read(4))[0]
        alen = struct.unpack("<Q", f.read(8))[0]
        for _ in range(alen):
            skip_gguf_value(f, etype)
    elif vtype in (10, 11, 12):
        f.read(8)


#只扫metadata，权重区不用读
def read_gguf_metadata(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise SystemExit("not a GGUF file")
        struct.unpack("<I", f.read(4))  # version，这里不校验
        struct.unpack("<Q", f.read(8))  # n_tensors，跳过
        n_metadata = struct.unpack("<Q", f.read(8))[0]

        tokens = None
        merges = None
        bos_id = None
        eos_id = None
        unk_id = None
        pad_id = None
        chat_template = None

        #每条：key字符串 + type + value；只要分词器相关字段
        for _ in range(n_metadata):
            key = read_gguf_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]

            #数组：先elem_type再长度。9是array，8是string
            if key == "tokenizer.ggml.tokens" and vtype == 9:
                struct.unpack("<I", f.read(4))
                alen = struct.unpack("<Q", f.read(8))[0]
                tokens = [read_gguf_string(f) for _ in range(alen)]
            elif key == "tokenizer.ggml.merges" and vtype == 9:
                struct.unpack("<I", f.read(4))
                alen = struct.unpack("<Q", f.read(8))[0]
                merges = [read_gguf_string(f) for _ in range(alen)]
            #4是uint32，bos/eos/unk/pad都是这种
            elif key == "tokenizer.ggml.bos_token_id" and vtype == 4:
                bos_id = struct.unpack("<I", f.read(4))[0]
            elif key == "tokenizer.ggml.eos_token_id" and vtype == 4:
                eos_id = struct.unpack("<I", f.read(4))[0]
            elif key == "tokenizer.ggml.unknown_token_id" and vtype == 4:
                unk_id = struct.unpack("<I", f.read(4))[0]
            elif key == "tokenizer.ggml.padding_token_id" and vtype == 4:
                pad_id = struct.unpack("<I", f.read(4))[0]
            elif key == "tokenizer.chat_template" and vtype == 8:
                chat_template = read_gguf_string(f)
            else:
                skip_gguf_value(f, vtype)

    return {
        "tokens": tokens,
        "merges": merges,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "unk_id": unk_id,
        "pad_id": pad_id,
        "chat_template": chat_template,
    }


def main():
    parser = argparse.ArgumentParser(description="导出GGUF里的tokenizer")
    parser.add_argument("gguf", help="输入的.gguf")
    parser.add_argument("--out-dir", default="models/chat")
    args = parser.parse_args()

    print(f"Reading tokenizer from {args.gguf}")
    meta = read_gguf_metadata(args.gguf)
    os.makedirs(args.out_dir, exist_ok=True)

    #下标就是token id，和GGUF里tokens数组顺序一致
    if meta["tokens"]:
        vocab = {token: i for i, token in enumerate(meta["tokens"])}
        vocab_path = os.path.join(args.out_dir, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, separators=(",", ":"))
        print(f"  Wrote {len(vocab)} entries to {vocab_path}")

    #一行一对，越靠前rank越小，BpeTokenizer按这个顺序合并
    if meta["merges"]:
        merges_path = os.path.join(args.out_dir, "merges.txt")
        with open(merges_path, "w", encoding="utf-8") as f:
            for rule in meta["merges"]:
                f.write(rule + "\n")
        print(f"  Wrote {len(meta['merges'])} merges to {merges_path}")

    #缺的id写成-1，和special_tokens.json的读法对齐
    special = {
        "bos_id": -1 if meta["bos_id"] is None else meta["bos_id"],
        "eos_id": -1 if meta["eos_id"] is None else meta["eos_id"],
        "unk_id": -1 if meta["unk_id"] is None else meta["unk_id"],
        "pad_id": -1 if meta["pad_id"] is None else meta["pad_id"],
    }
    special_path = os.path.join(args.out_dir, "special_tokens.json")
    with open(special_path, "w", encoding="utf-8") as f:
        json.dump(special, f, indent=2)
        f.write("\n")
    print(f"  Wrote special tokens to {special_path}")

    #聊天模板是可选的，没有就跳过
    if meta["chat_template"]:
        template_path = os.path.join(args.out_dir, "chat_template.txt")
        with open(template_path, "w", encoding="utf-8") as f:
            f.write(meta["chat_template"])
        print(f"  Wrote chat template to {template_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
