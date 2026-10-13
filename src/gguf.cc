//GGUF v3底层读取：按小端把header / metadata / tensor info拆出来
//并算出data区起点；真正装进MiniLlamaModel，留给gguf_loader
//文件顺序：Header -> Metadata KV -> Tensor Info -> padding -> Tensor Data
#include "mini_llama/gguf.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mini_llama {

//权重数据类型名字，inspect打印用
const char* GgmlTypeName(uint32_t type) {
  switch (type) {
    case kGgmlTypeF32:
      return "F32";
    case kGgmlTypeF16:
      return "F16";
    case kGgmlTypeQ40:
      return "Q4_0";
    case kGgmlTypeQ41:
      return "Q4_1";
    case kGgmlTypeQ80:
      return "Q8_0";
    default:
      return "UNKNOWN";
  }
}

//metadata值类型名字，inspect打印用
const char* GgufValueTypeName(uint32_t type) {
  switch (type) {
    case kGgufTypeUint8:
      return "uint8";
    case kGgufTypeInt8:
      return "int8";
    case kGgufTypeUint16:
      return "uint16";
    case kGgufTypeInt16:
      return "int16";
    case kGgufTypeUint32:
      return "uint32";
    case kGgufTypeInt32:
      return "int32";
    case kGgufTypeFloat32:
      return "float32";
    case kGgufTypeBool:
      return "bool";
    case kGgufTypeString:
      return "string";
    case kGgufTypeArray:
      return "array";
    case kGgufTypeUint64:
      return "uint64";
    case kGgufTypeInt64:
      return "int64";
    case kGgufTypeFloat64:
      return "float64";
    default:
      return "unknown";
  }
}

namespace {

//防坏文件把内存撑爆的上限
constexpr uint64_t kDefaultAlignment = 32;//没写general.alignment时用这个
constexpr uint64_t kMaxAlignment = 1024 * 1024;
constexpr uint64_t kMaxStringBytes = 16 * 1024 * 1024;
constexpr uint64_t kMaxMetadataEntries = 1000000;
constexpr uint64_t kMaxTensorEntries = 1000000;
constexpr uint64_t kMaxArrayElements = 10000000;
constexpr uint32_t kMaxTensorDims = 8;
constexpr uint64_t kQuantBlockSize = 32;//量化按32个元素一块

//小端读固定宽度整数，浮点数；x86/Windows本机就是小端，直接memcpy
template <typename T>
bool ReadLe(std::ifstream& f, T& out) {
  char buf[sizeof(T)];
  if (!f.read(buf, sizeof(T))) {
    return false;
  }
  std::memcpy(&out, buf, sizeof(T));
  return true;
}

//读小端uint32 ， uint64的包装
bool ReadU32(std::ifstream& f, uint32_t& out) { return ReadLe(f, out); }

bool ReadU64(std::ifstream& f, uint64_t& out) { return ReadLe(f, out); }

//GGUF字符串：先uint64长度，再跟这么多字节
bool ReadString(std::ifstream& f, std::string& out) {
  uint64_t len = 0;
  if (!ReadU64(f, len)) {
    return false;
  }
  //判断长度
  if (len > kMaxStringBytes ||
      len >
          static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
    return false;
  }
  out.resize(len);
  if (len > 0 && !f.read(out.data(), static_cast<std::streamsize>(len))) {
    return false;
  }
  return true;
}

//跳过一个value：找metadata时用，只往前挪文件指针，不需要关心内容
bool SkipValue(std::ifstream& f, uint32_t type) {
  switch (type) {
    case kGgufTypeUint8:
    case kGgufTypeInt8: {
      char dummy;
      return f.read(&dummy, 1).good();
    }
    case kGgufTypeUint16:
    case kGgufTypeInt16: {
      char dummy[2];
      return f.read(dummy, 2).good();
    }
    case kGgufTypeUint32:
    case kGgufTypeInt32:
    case kGgufTypeFloat32: {
      char dummy[4];
      return f.read(dummy, 4).good();
    }
    case kGgufTypeBool: {
      char dummy;
      return f.read(&dummy, 1).good();
    }
    case kGgufTypeString: {
      std::string dummy;
      return ReadString(f, dummy);
    }
    case kGgufTypeUint64:
    case kGgufTypeInt64:
    case kGgufTypeFloat64: {
      char dummy[8];
      return f.read(dummy, 8).good();
    }
    case kGgufTypeArray: {
      //array：elem_type+len+每个元素
      uint32_t elem_type = 0;
      uint64_t len = 0;
      if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
        return false;
      }
      if (len > kMaxArrayElements) {
        return false;
      }
      for (uint64_t i = 0; i < len; ++i) {
        if (!SkipValue(f, elem_type)) {
          return false;
        }
      }
      return true;
    }
    default:
      return false;
  }
}

//先检查再乘，避免溢出
bool MultiplyChecked(uint64_t a, uint64_t b, uint64_t& out) {
  if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
    return false;
  }
  out = a * b;
  return true;
}

//先检查再加，算绝对offset时避免溢出
bool AddChecked(uint64_t a, uint64_t b, uint64_t& out) {
  if (b > std::numeric_limits<uint64_t>::max() - a) {
    return false;
  }
  out = a + b;
  return true;
}

//元素个数 = 各维相乘，任一维为0直接失败
bool TensorNumel(const GgufTensorInfo& tensor, uint64_t& out) {
  out = 1;
  for (uint64_t dim : tensor.shape) {
    if (dim == 0) {
      return false;
    }
    if (!MultiplyChecked(out, dim, out)) {
      return false;
    }
  }
  return true;
}

//按GGML类型算这块权重在data区占多少字节
//F32/F16按元素算；量化按block算（一块32个元素）
bool TensorDataSize(const GgufTensorInfo& tensor, uint64_t& out) {
  uint64_t num_elements = 0;
  if (!TensorNumel(tensor, num_elements)) {
    return false;
  }

  switch (tensor.type) {
    case kGgmlTypeF32:
      return MultiplyChecked(num_elements, 4, out);
    case kGgmlTypeF16:
      return MultiplyChecked(num_elements, 2, out);
    case kGgmlTypeQ40: {
      if (num_elements % kQuantBlockSize != 0) {
        return false;
      }
      uint64_t n_blocks = num_elements / kQuantBlockSize;
      return MultiplyChecked(n_blocks, 18, out);//Q4_0一块18字节
    }
    case kGgmlTypeQ41: {
      if (num_elements % kQuantBlockSize != 0) {
        return false;
      }
      uint64_t n_blocks = num_elements / kQuantBlockSize;
      return MultiplyChecked(n_blocks, 20, out);//Q4_1一块20字节
    }
    case kGgmlTypeQ80: {
      if (num_elements % kQuantBlockSize != 0) {
        return false;
      }
      uint64_t n_blocks = num_elements / kQuantBlockSize;
      return MultiplyChecked(n_blocks, 34, out);//Q8_0一块34字节
    }
    default:
      out = 0;
      return true;
  }
}

//读一个scalar/array，转成可看的字符串，inspect打印用
//array太大只打部分，比如[128 x string]，元素本身skip掉
bool ReadValueAsString(std::ifstream& f, uint32_t type, std::string& out) {
  std::ostringstream oss;
  switch (type) {
    case kGgufTypeUint8: {
      uint8_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << static_cast<unsigned>(v);
      break;
    }
    case kGgufTypeInt8: {
      int8_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << static_cast<int>(v);
      break;
    }
    case kGgufTypeUint16: {
      uint16_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << v;
      break;
    }
    case kGgufTypeInt16: {
      int16_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << v;
      break;
    }
    case kGgufTypeUint32: {
      uint32_t v = 0;
      if (!ReadU32(f, v)) {
        return false;
      }
      oss << v;
      break;
    }
    case kGgufTypeInt32: {
      int32_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << v;
      break;
    }
    case kGgufTypeFloat32: {
      float v = 0.0f;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << v;
      break;
    }
    case kGgufTypeBool: {
      uint8_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << (v ? "true" : "false");
      break;
    }
    case kGgufTypeString: {
      if (!ReadString(f, out)) {
        return false;
      }
      return true;
    }
    case kGgufTypeUint64: {
      uint64_t v = 0;
      if (!ReadU64(f, v)) {
        return false;
      }
      oss << v;
      break;
    }
    case kGgufTypeInt64: {
      int64_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << v;
      break;
    }
    case kGgufTypeFloat64: {
      double v = 0.0;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << v;
      break;
    }
    case kGgufTypeArray: {
      //大数组只打印部分，元素本身skip掉
      uint32_t elem_type = 0;
      uint64_t len = 0;
      if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
        return false;
      }
      if (len > kMaxArrayElements) {
        return false;
      }
      oss << "[" << len << " x " << GgufValueTypeName(elem_type) << "]";
      for (uint64_t i = 0; i < len; ++i) {
        if (!SkipValue(f, elem_type)) {
          return false;
        }
      }
      break;
    }
    default:
      return false;
  }
  out = oss.str();
  return true;
}

//从metadata里取alignment；没有就默认32；必须是2的幂，否则返回0表示非法
uint64_t ParseAlignment(const std::vector<GgufMetadataKv>& metadata) {
  uint64_t alignment = kDefaultAlignment;
  for (const auto& kv : metadata) {
    if (kv.key != "general.alignment") {
      continue;
    }
    try {
      size_t parsed = 0;
      alignment = std::stoull(kv.value_str, &parsed);
      if (parsed != kv.value_str.size()) {
        return 0;//尾巴粘了别的东西，不算合法数字
      }
    } catch (const std::exception&) {
      return 0;
    }
  }
  if (alignment == 0 || alignment > kMaxAlignment) {
    return 0;
  }
  //2的幂判定：alignment & (alignment-1) == 0；48这种会挂
  if ((alignment & (alignment - 1)) != 0) {
    return 0;
  }
  return alignment;
}

//打开文件，扫metadata找到key；成功时指针停在type后面、value前面
//给GgufGetMetadata*用，不用整份Load也能按名取值
bool OpenAndFindMetadataKey(std::ifstream& f, const std::string& path,
                            const std::string& key, uint32_t& out_type) {
  f.open(path, std::ios::binary);
  if (!f.is_open()) {
    return false;
  }
  //跳过magic(4)+version(4)，落到n_tensors
  f.seekg(8, std::ios::beg);
  if (!f.good()) {
    return false;
  }

  uint64_t n_tensors = 0;
  uint64_t n_metadata = 0;
  if (!ReadU64(f, n_tensors) || !ReadU64(f, n_metadata)) {
    return false;
  }

  for (uint64_t i = 0; i < n_metadata; ++i) {
    std::string read_key;
    if (!ReadString(f, read_key)) {
      return false;
    }
    uint32_t type = 0;
    if (!ReadU32(f, type)) {
      return false;
    }
    if (read_key == key) {
      out_type = type;
      return true;//找到了，调用方接着读value
    }
    //不是要找的key，跳过value继续往后扫
    if (!SkipValue(f, type)) {
      return false;
    }
  }
  return false;//扫完也没有
}

}

//解析整份GGUF：只拆结构+校验边界，不把权重解码进模型
bool GgufReader::Load(const std::string& path) {
  //每次Load先清状态，方便重复调用（第二次失败时旧数据不能留着）
  version = 0;
  n_tensors = 0;
  n_metadata = 0;
  metadata.clear();
  tensors.clear();
  data_offset = 0;
  load_error.clear();

  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    load_error = "failed to open: " + path;
    return false;
  }

  //----- 1) Header -----
  //magic四个字节必须是"GGUF"，否则根本不是这种文件
  char magic[4];
  if (!f.read(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
    load_error = "invalid GGUF magic";
    return false;
  }

  if (!ReadU32(f, version)) {
    load_error = "failed to read version";
    return false;
  }
  //学习版只认v3
  if (version != 3) {
    load_error = "unsupported GGUF version: " + std::to_string(version);
    return false;
  }

  if (!ReadU64(f, n_tensors) || !ReadU64(f, n_metadata)) {
    load_error = "failed to read header counts";
    return false;
  }
  if (n_tensors > kMaxTensorEntries) {
    load_error = "too many tensors: " + std::to_string(n_tensors);
    return false;
  }
  if (n_metadata > kMaxMetadataEntries) {
    load_error = "too many metadata entries: " + std::to_string(n_metadata);
    return false;
  }

  //----- 2) Metadata：一串KV，描述架构名、对齐等 -----
  //每条：key + type + value；value先收成字符串方便inspect
  metadata.reserve(n_metadata);
  std::unordered_set<std::string> metadata_keys;
  for (uint64_t i = 0; i < n_metadata; ++i) {
    GgufMetadataKv kv;
    if (!ReadString(f, kv.key)) {
      load_error = "failed to read metadata key at index " + std::to_string(i);
      return false;
    }
    //key不能重，否则后面按名取会搞不清用哪个
    if (!metadata_keys.insert(kv.key).second) {
      load_error = "duplicate metadata key: " + kv.key;
      return false;
    }
    if (!ReadU32(f, kv.type)) {
      load_error = "failed to read metadata type for key: " + kv.key;
      return false;
    }
    if (!ReadValueAsString(f, kv.type, kv.value_str)) {
      load_error = "failed to read metadata value for key: " + kv.key;
      return false;
    }
    metadata.push_back(std::move(kv));
  }

  //----- 3) Tensor info：目录，还没读权重字节 -----
  //每条：name + n_dims + shape[] + ggml type + 相对data起点的offset
  tensors.reserve(n_tensors);
  for (uint64_t i = 0; i < n_tensors; ++i) {
    GgufTensorInfo info;
    if (!ReadString(f, info.name)) {
      load_error = "failed to read tensor name at index " + std::to_string(i);
      return false;
    }
    uint32_t n_dims = 0;
    if (!ReadU32(f, n_dims)) {
      load_error = "failed to read tensor n_dims for: " + info.name;
      return false;
    }
    if (n_dims == 0 || n_dims > kMaxTensorDims) {
      load_error = "invalid tensor n_dims for: " + info.name;
      return false;
    }
    info.shape.resize(n_dims);
    for (uint32_t d = 0; d < n_dims; ++d) {
      if (!ReadU64(f, info.shape[d])) {
        load_error = "failed to read tensor shape for: " + info.name;
        return false;
      }
      //维数不能为0，否则numel算出来没意义
      if (info.shape[d] == 0) {
        load_error = "invalid zero tensor dimension for: " + info.name;
        return false;
      }
    }
    if (!ReadU32(f, info.type)) {
      load_error = "failed to read tensor type for: " + info.name;
      return false;
    }
    //注意：这里的offset是相对data区，不是文件开头
    if (!ReadU64(f, info.offset)) {
      load_error = "failed to read tensor offset for: " + info.name;
      return false;
    }
    tensors.push_back(std::move(info));
  }

  //tensor名字也不能重
  std::unordered_set<std::string> names;
  for (const auto& tensor : tensors) {
    if (!names.insert(tensor.name).second) {
      load_error = "duplicate tensor name: " + tensor.name;
      return false;
    }
  }

  //----- 4) Data区起点 = 当前文件位置，再按alignment向上取整 -----
  std::streampos current = f.tellg();
  if (current < 0) {
    load_error = "failed to determine GGUF data offset";
    return false;
  }
  data_offset = static_cast<uint64_t>(current);
  uint64_t alignment = ParseAlignment(metadata);
  if (alignment == 0) {
    load_error = "invalid general.alignment";
    return false;
  }
  //padding让data_offset变成alignment的倍数
  uint64_t padding = (alignment - (data_offset % alignment)) % alignment;
  data_offset += padding;

  //每块tensor的绝对区间不能超出文件末尾
  f.seekg(0, std::ios::end);
  std::streampos end = f.tellg();
  if (end < 0 || static_cast<uint64_t>(end) < data_offset) {
    load_error = "GGUF file ended before tensor data section";
    return false;
  }
  uint64_t file_size = static_cast<uint64_t>(end);

  for (const auto& tensor : tensors) {
    uint64_t tensor_size = 0;
    if (!TensorDataSize(tensor, tensor_size)) {
      load_error = "invalid tensor data size for: " + tensor.name;
      return false;
    }
    if (tensor_size == 0) {
      continue;//未知类型先跳过边界检查
    }

    //绝对起点 = data_offset + 相对offset
    uint64_t tensor_start = 0;
    if (!AddChecked(data_offset, tensor.offset, tensor_start)) {
      load_error = "tensor offset overflow for: " + tensor.name;
      return false;
    }
    uint64_t tensor_end = 0;
    if (!AddChecked(tensor_start, tensor_size, tensor_end)) {
      load_error = "tensor data range overflow for: " + tensor.name;
      return false;
    }
    if (tensor_end > file_size) {
      load_error = "tensor data outside file bounds: " + tensor.name;
      return false;
    }
  }

  return true;
}

//把Load出来的结构打到终端，给inspect-gguf用
//顺序：文件摘要 -> metadata列表 -> tensor名字/形状（不打印权重数值）
void InspectGguf(const GgufReader& reader) {
  //先打header级摘要
  std::cout << "=== GGUF Info ===" << std::endl;
  std::cout << "  version:     " << reader.version << std::endl;
  std::cout << "  n_tensors:   " << reader.n_tensors << std::endl;
  std::cout << "  n_metadata:  " << reader.n_metadata << std::endl;
  std::cout << "  data_offset: " << reader.data_offset << std::endl;

  //再打每条KV
  if (!reader.metadata.empty()) {
    std::cout << std::endl;
    std::cout << "=== Metadata ===" << std::endl;
    for (const auto& kv : reader.metadata) {
      std::cout << "  " << kv.key << " (" << GgufValueTypeName(kv.type)
                << "): " << kv.value_str << std::endl;
    }
  }

  //最后打tensor目录；offset仍是相对data区的
  if (!reader.tensors.empty()) {
    std::cout << std::endl;
    std::cout << "=== Tensors ===" << std::endl;
    for (const auto& t : reader.tensors) {
      std::cout << "  " << t.name << std::endl;
      std::cout << "    shape:  [";
      for (size_t i = 0; i < t.shape.size(); ++i) {
        if (i > 0) {
          std::cout << ", ";
        }
        std::cout << t.shape[i];
      }
      std::cout << "]" << std::endl;
      std::cout << "    dtype:  " << GgmlTypeName(t.type) << " (" << t.type
                << ")" << std::endl;
      std::cout << "    offset: " << t.offset << std::endl;
    }
  }
}

//对外暴露的字节数计算，给后面gguf_loader用
bool GgufTensorDataSize(const GgufTensorInfo& info, uint64_t& out_size) {
  return TensorDataSize(info, out_size);
}

//从data区读出一块原始字节；失败返回空vector
//绝对位置 = data_offset + info.offset
std::vector<uint8_t> ReadGgufTensorRaw(const std::string& path,
                                       const GgufTensorInfo& info,
                                       uint64_t data_offset) {
  std::vector<uint8_t> result;
  uint64_t tensor_size = 0;
  if (!TensorDataSize(info, tensor_size) || tensor_size == 0) {
    return result;
  }

  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    return result;
  }

  //拼出文件里的绝对起点再seek
  uint64_t tensor_start = 0;
  if (!AddChecked(data_offset, info.offset, tensor_start)) {
    return result;
  }
  if (tensor_start >
      static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
    return result;
  }

  f.seekg(static_cast<std::streamoff>(tensor_start), std::ios::beg);
  if (!f.good()) {
    return result;
  }

  result.resize(tensor_size);
  f.read(reinterpret_cast<char*>(result.data()),
         static_cast<std::streamsize>(tensor_size));
  //读少了说明文件截断了，清空当失败
  if (static_cast<uint64_t>(f.gcount()) != tensor_size) {
    result.clear();
  }
  return result;
}

//下面几个GgufGetMetadata*：按key取类型化的值，给后面映射config用
//会重新打开文件扫metadata，不用先Load整份

//读整数类metadata（uint8/int32/uint64等都收成int64）
bool GgufGetMetadataInt(const std::string& path, const std::string& key,
                        int64_t& out) {
  std::ifstream f;
  uint32_t type = 0;
  if (!OpenAndFindMetadataKey(f, path, key, type)) {
    return false;
  }
  switch (type) {
    case kGgufTypeUint8: {
      uint8_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      out = static_cast<int64_t>(v);
      return true;
    }
    case kGgufTypeInt8: {
      int8_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      out = v;
      return true;
    }
    case kGgufTypeUint16: {
      uint16_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      out = static_cast<int64_t>(v);
      return true;
    }
    case kGgufTypeInt16: {
      int16_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      out = v;
      return true;
    }
    case kGgufTypeUint32: {
      uint32_t v = 0;
      if (!ReadU32(f, v)) {
        return false;
      }
      out = static_cast<int64_t>(v);
      return true;
    }
    case kGgufTypeInt32: {
      int32_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      out = v;
      return true;
    }
    case kGgufTypeUint64: {
      uint64_t v = 0;
      if (!ReadU64(f, v)) {
        return false;
      }
      out = static_cast<int64_t>(v);
      return true;
    }
    case kGgufTypeInt64: {
      int64_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      out = v;
      return true;
    }
    default:
      return false;
  }
}

//读浮点metadata（float32/float64都收成double）
bool GgufGetMetadataFloat(const std::string& path, const std::string& key,
                          double& out) {
  std::ifstream f;
  uint32_t type = 0;
  if (!OpenAndFindMetadataKey(f, path, key, type)) {
    return false;
  }
  switch (type) {
    case kGgufTypeFloat32: {
      float v = 0.0f;
      if (!ReadLe(f, v)) {
        return false;
      }
      out = static_cast<double>(v);
      return true;
    }
    case kGgufTypeFloat64: {
      double v = 0.0;
      if (!ReadLe(f, v)) {
        return false;
      }
      out = v;
      return true;
    }
    default:
      return false;
  }
}

//读字符串metadata，类型必须刚好是STRING（比如general.architecture）
bool GgufGetMetadataString(const std::string& path, const std::string& key,
                           std::string& out) {
  std::ifstream f;
  uint32_t type = 0;
  if (!OpenAndFindMetadataKey(f, path, key, type)) {
    return false;
  }
  if (type != kGgufTypeString) {
    return false;
  }
  return ReadString(f, out);
}

//读字符串数组，比如tokenizer.ggml.tokens词表
bool GgufGetMetadataStringArray(const std::string& path, const std::string& key,
                                std::vector<std::string>& out) {
  std::ifstream f;
  uint32_t type = 0;
  if (!OpenAndFindMetadataKey(f, path, key, type)) {
    return false;
  }
  if (type != kGgufTypeArray) {
    return false;
  }
  uint32_t elem_type = 0;
  uint64_t len = 0;
  if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
    return false;
  }
  if (elem_type != kGgufTypeString || len > kMaxArrayElements) {
    return false;
  }
  out.clear();
  out.reserve(len);
  for (uint64_t i = 0; i < len; ++i) {
    std::string s;
    if (!ReadString(f, s)) {
      return false;
    }
    out.push_back(std::move(s));
  }
  return true;
}

//读int32数组，比如tokenizer.ggml.token_type
bool GgufGetMetadataIntArray(const std::string& path, const std::string& key,
                             std::vector<int32_t>& out) {
  std::ifstream f;
  uint32_t type = 0;
  if (!OpenAndFindMetadataKey(f, path, key, type)) {
    return false;
  }
  if (type != kGgufTypeArray) {
    return false;
  }
  uint32_t elem_type = 0;
  uint64_t len = 0;
  if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
    return false;
  }
  if (elem_type != kGgufTypeInt32 || len > kMaxArrayElements) {
    return false;
  }
  out.clear();
  out.reserve(len);
  for (uint64_t i = 0; i < len; ++i) {
    int32_t v = 0;
    if (!ReadLe(f, v)) {
      return false;
    }
    out.push_back(v);
  }
  return true;
}

}
