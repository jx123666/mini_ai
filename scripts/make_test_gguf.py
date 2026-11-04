#!/usr/bin/env python3
#生成最小GGUF v3测试文件，给inspect-gguf / test_gguf用
#Header -> Metadata(3条) -> Tensor Info(1个) -> padding对齐 -> Tensor Data(全0)
#"<I"小端uint32，"<Q"小端uint64，和C++ reader按同样字节序读

import argparse
import os
import struct


#GGUF字符串：先写小端uint64长度，再写utf-8字节（不是\0结尾）
def write_string(buf: bytearray, s: str):
    b = s.encode("utf-8")
    buf.extend(struct.pack("<Q", len(b)))
    buf.extend(b)


def make_test_gguf(output_path: str):
    #权重数据区：128*32个float32，全0就够测读取和对齐，不用真权重
    n_floats = 128 * 32
    tensor_data = bytes(4 * n_floats)

    #先在内存里拼header+metadata+tensor info，方便算data起点和padding
    info = bytearray()

    # Header
    #magic四个字节必须是GGUF，reader靠这个认文件
    info.extend(b"GGUF")
    #version=3，本项目只支持v3
    info.extend(struct.pack("<I", 3))
    #告诉后面：有1个tensor、3条metadata
    info.extend(struct.pack("<Q", 1))
    info.extend(struct.pack("<Q", 3))

    # Metadata KV
    #每条：key字符串 + type(uint32) + value
    #type编号见gguf.h：8=STRING，4=UINT32

    #架构名
    write_string(info, "general.architecture")
    info.extend(struct.pack("<I", 8))
    write_string(info, "llama")

    #模型显示名
    write_string(info, "general.name")
    info.extend(struct.pack("<I", 8))
    write_string(info, "tiny-test")

    #数据区对齐；必须是2的幂，reader用它算data_offset
    write_string(info, "general.alignment")
    info.extend(struct.pack("<I", 4))
    info.extend(struct.pack("<I", 64))

    #Tensor Info（目录，还不是权重本身）
    #格式：name+n_dims+各维大小+ggml type + 相对data起点的offset
    write_string(info, "token_embd.weight")
    info.extend(struct.pack("<I", 2))  #两维
    info.extend(struct.pack("<Q", 128))  #dim0
    info.extend(struct.pack("<Q", 32))  #dim1
    info.extend(struct.pack("<I", 0))  #0=F32
    info.extend(struct.pack("<Q", 0))  #相对offset=0，紧挨data起点

    #对齐到data区
    #info长度不是64的倍数时，中间填padding，再写真正的权重字节
    alignment = 64
    data_start = len(info)
    padding = (alignment - (data_start % alignment)) % alignment

    with open(output_path, "wb") as f:
        f.write(info)
        f.write(b"\x00" * padding)
        f.write(tensor_data)

    #打一下各段大小，方便对照reader算出的data_offset
    print(f"Wrote {output_path}")
    print(f"  info size: {data_start} bytes")
    print(f"  padding: {padding} bytes")
    print(f"  data size: {len(tensor_data)} bytes")
    print(f"  total: {data_start + padding + len(tensor_data)} bytes")


def main():
    parser = argparse.ArgumentParser(
        description="生成最小GGUF测试文件（header+metadata+一个F32 tensor）"
    )
    parser.add_argument("--output", type=str, default="models/tiny/test.gguf")
    args = parser.parse_args()
    #输出目录不存在就建一下
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    make_test_gguf(args.output)


if __name__ == "__main__":
    main()

