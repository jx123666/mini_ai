//GGUF底层读取：只负责拆文件结构header/metadata/tensor info/data，不做映射
#ifndef INCLUDE_MINI_LLAMA_GGUF_H_
#define INCLUDE_MINI_LLAMA_GGUF_H_

#include <cstdint>
#include <string>
#include <vector>

namespace mini_llama {

//GGML张量数据类型，仅支持float32、float16 以及压缩格式q4_0、q4_1、q8_0
enum GgmlType : uint32_t {
  kGgmlTypeF32 = 0,
  kGgmlTypeF16 = 1,
  kGgmlTypeQ40 = 2,
  kGgmlTypeQ41 = 3,
  kGgmlTypeQ80 = 8,
};

const char* GgmlTypeName(uint32_t type);

//GGUF metadata值类型
enum GgufValueType : uint32_t {
  kGgufTypeUint8 = 0,
  kGgufTypeInt8 = 1,
  kGgufTypeUint16 = 2,
  kGgufTypeInt16 = 3,
  kGgufTypeUint32 = 4,
  kGgufTypeInt32 = 5,
  kGgufTypeFloat32 = 6,
  kGgufTypeBool = 7,
  kGgufTypeString = 8,
  kGgufTypeArray = 9,
  kGgufTypeUint64 = 10,
  kGgufTypeInt64 = 11,
  kGgufTypeFloat64 = 12,
};

const char* GgufValueTypeName(uint32_t type);

//tensor info，名字、形状、类型、相对数据区的offset
struct GgufTensorInfo {
  std::string name;
  std::vector<uint64_t> shape;
  uint32_t type = 0;
  uint64_t offset = 0;
};

//一条metadata：key + 类型 + 给人看的字符串值
struct GgufMetadataKv {
  std::string key;
  uint32_t type = 0;
  std::string value_str;
};

//最小GGUF v3读取器
struct GgufReader {
  uint32_t version = 0;
  uint64_t n_tensors = 0;
  uint64_t n_metadata = 0;
  std::vector<GgufMetadataKv> metadata;
  std::vector<GgufTensorInfo> tensors;
  uint64_t data_offset = 0;//tensor data区在文件里的绝对起点

  //解析文件；失败返回false，原因在load_error
  bool Load(const std::string& path);

  std::string load_error;
};

//打印metadata和tensor名字，给inspect-gguf用
void InspectGguf(const GgufReader& reader);

//按类型和shape算这块tensor数据占多少字节
bool GgufTensorDataSize(const GgufTensorInfo& info, uint64_t& out_size);

//从data区读出一块原始字节
std::vector<uint8_t> ReadGgufTensorRaw(const std::string& path,
                                       const GgufTensorInfo& info,
                                       uint64_t data_offset);

//按key读类型化的metadata，会重新打开文件找
bool GgufGetMetadataInt(const std::string& path, const std::string& key,
                        int64_t& out);
bool GgufGetMetadataBool(const std::string& path, const std::string& key,
                         bool& out);
bool GgufGetMetadataFloat(const std::string& path, const std::string& key,
                          double& out);
bool GgufGetMetadataString(const std::string& path, const std::string& key,
                           std::string& out);
bool GgufGetMetadataStringArray(const std::string& path, const std::string& key,
                                std::vector<std::string>& out);
bool GgufGetMetadataIntArray(const std::string& path, const std::string& key,
                             std::vector<int32_t>& out);

}

#endif
