//一次用户请求的追踪：耗时、token 数、阶段事件、错误
//跟推理 Context 不是一回事，这个更偏日志
#ifndef INCLUDE_MINI_LLAMA_REQUEST_CONTEXT_H_
#define INCLUDE_MINI_LLAMA_REQUEST_CONTEXT_H_

#include <chrono>
#include <string>
#include <vector>

namespace mini_llama {

using RequestClock = std::chrono::steady_clock;

//一个阶段事件，比如 tokenize / prefill / decode
struct TraceEvent {
  std::string stage;
  double elapsed_ms = 0.0;
  int tokens = 0;
  std::string detail;
};

//请求级上下文：从进门到结束记一圈
struct RequestContext {
  std::string trace_id;
  std::string mode;
  std::string backend;
  std::string model_path;

  RequestClock::time_point start_time;
  double total_ms = 0.0;

  int prompt_tokens = 0;
  int prefill_tokens = 0;
  int decode_tokens = 0;
  int generated_tokens = 0;

  double model_load_ms = 0.0;
  double tokenize_ms = 0.0;
  double prefill_ms = 0.0;
  double decode_ms = 0.0;
  double sample_ms = 0.0;

  std::string error;
  std::vector<TraceEvent> events;

  void Start();
  void Finish();
  void RecordEvent(const std::string& stage, double elapsed_ms, int tokens = 0,
                   const std::string& detail = "");
  void SetError(const std::string& message);
  bool ok() const;//没有 error 就算 ok
};

std::string NewTraceId();
double ElapsedMs(RequestClock::time_point start);
RequestContext StartRequest(const std::string& mode, const std::string& backend,
                            const std::string& model_path);
std::string FormatRequestTraceSummary(const RequestContext& request);
std::vector<std::string> FormatRequestTraceEvents(
    const RequestContext& request);

}

#endif
