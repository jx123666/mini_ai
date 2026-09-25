#include "mini_llama/request_context.h"

#include <atomic>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace mini_llama {
namespace {

std::atomic<unsigned long long> g_request_counter{0};

//格式化成三位小数
std::string FormatMs(double value) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(3) << value;
  return out.str();
}

}

//开始计时
void RequestContext::Start() { start_time = RequestClock::now(); }

//结束时算出总耗时
void RequestContext::Finish() { total_ms = ElapsedMs(start_time); }

//记一个阶段事件
void RequestContext::RecordEvent(const std::string& stage, double elapsed_ms,
                                 int tokens, const std::string& detail) {
  events.push_back({stage, elapsed_ms, tokens, detail});
}

void RequestContext::SetError(const std::string& message) { error = message; }

bool RequestContext::ok() const { return error.empty(); }

//生成唯一的trace_id
std::string NewTraceId() {
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  const unsigned long long seq = ++g_request_counter;

  std::tm tm_snapshot{};
#if defined(_WIN32)
  localtime_s(&tm_snapshot, &time);
#else
  localtime_r(&time, &tm_snapshot);
#endif

  std::ostringstream out;
  out << "req_" << std::put_time(&tm_snapshot, "%Y%m%d%H%M%S") << "_"
      << std::setw(6) << std::setfill('0') << seq;
  return out.str();
}

//从start到现在过了多少毫秒
double ElapsedMs(RequestClock::time_point start) {
  return std::chrono::duration<double, std::milli>(RequestClock::now() - start)
      .count();
}

//开一个新请求并开始计时
RequestContext StartRequest(const std::string& mode, const std::string& backend,
                            const std::string& model_path) {
  RequestContext request;
  request.trace_id = NewTraceId();
  request.mode = mode;
  request.backend = backend;
  request.model_path = model_path;
  request.Start();
  return request;
}

//拼一条摘要日志
std::string FormatRequestTraceSummary(const RequestContext& request) {
  std::ostringstream out;
  out << "trace summary: trace_id=" << request.trace_id
      << " mode=" << request.mode
      << " status=" << (request.ok() ? "ok" : "error")
      << " backend=" << request.backend
      << " model=" << request.model_path
      << " prompt_tokens=" << request.prompt_tokens
      << " prefill_tokens=" << request.prefill_tokens
      << " decode_tokens=" << request.decode_tokens
      << " generated_tokens=" << request.generated_tokens
      << " model_load_ms=" << FormatMs(request.model_load_ms)
      << " tokenize_ms=" << FormatMs(request.tokenize_ms)
      << " prefill_ms=" << FormatMs(request.prefill_ms)
      << " decode_ms=" << FormatMs(request.decode_ms)
      << " sample_ms=" << FormatMs(request.sample_ms)
      << " total_ms=" << FormatMs(request.total_ms);
  if (!request.error.empty()) {
    out << " error=\"" << request.error << "\"";
  }
  return out.str();
}

//把每个事件格式化成一行
std::vector<std::string> FormatRequestTraceEvents(
    const RequestContext& request) {
  std::vector<std::string> lines;
  lines.reserve(request.events.size());
  for (const TraceEvent& event : request.events) {
    std::ostringstream out;
    out << "trace event: trace_id=" << request.trace_id
        << " stage=" << event.stage << " tokens=" << event.tokens
        << " elapsed_ms=" << FormatMs(event.elapsed_ms);
    if (!event.detail.empty()) {
      out << " detail=\"" << event.detail << "\"";
    }
    lines.push_back(out.str());
  }
  return lines;
}

}
