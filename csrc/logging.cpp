#include "logging.h"

#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>

#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

namespace flexkv::logging {
namespace {

class UppercaseLevelFlag : public spdlog::custom_flag_formatter {
public:
  void format(const spdlog::details::log_msg &message, const std::tm &,
              spdlog::memory_buf_t &destination) override {
    const auto level = spdlog::level::to_string_view(message.level);
    for (const char ch : level) {
      destination.push_back(
          static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
    }
  }

  std::unique_ptr<custom_flag_formatter> clone() const override {
    return spdlog::details::make_unique<UppercaseLevelFlag>();
  }
};

spdlog::level::level_enum ToSpdlogLevel(Level level) noexcept {
  switch (level) {
  case Level::Debug:
    return spdlog::level::debug;
  case Level::Info:
    return spdlog::level::info;
  case Level::Warning:
    return spdlog::level::warn;
  case Level::Error:
    return spdlog::level::err;
  case Level::Critical:
    return spdlog::level::critical;
  }
  return spdlog::level::info;
}

spdlog::level::level_enum ParseLevel() {
  const char *configured = std::getenv("FLEXKV_LOG_LEVEL");
  if (configured == nullptr || configured[0] == '\0') {
    return spdlog::level::info;
  }

  std::string level(configured);
  for (char &ch : level) {
    ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
  }
  if (level == "DEBUG")
    return spdlog::level::debug;
  if (level == "INFO")
    return spdlog::level::info;
  if (level == "WARNING" || level == "WARN")
    return spdlog::level::warn;
  if (level == "ERROR")
    return spdlog::level::err;
  if (level == "CRITICAL")
    return spdlog::level::critical;
  if (level == "OFF")
    return spdlog::level::off;
  return spdlog::level::info;
}

std::string EscapePattern(std::string value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char ch : value) {
    if (ch == '%')
      escaped.push_back('%');
    escaped.push_back(ch);
  }
  return escaped;
}

std::shared_ptr<spdlog::logger> CreateLogger() {
  const char *configured_prefix = std::getenv("FLEXKV_LOGGING_PREFIX");
  const std::string prefix = EscapePattern(
      configured_prefix == nullptr || configured_prefix[0] == '\0'
          ? "FLEXKV"
          : configured_prefix);

  auto sink = std::make_shared<spdlog::sinks::stdout_sink_mt>();
  auto logger = std::make_shared<spdlog::logger>("flexkv_native", sink);
  auto formatter = std::make_unique<spdlog::pattern_formatter>();
  formatter->add_flag<UppercaseLevelFlag>('*').set_pattern(
      "[" + prefix +
      "] %* %m-%d %H:%M:%S.%e [pid=%P tid=%t] [%s:%#] "
      "[FlexKV-Native] %v");
  logger->set_formatter(std::move(formatter));
  logger->set_level(ParseLevel());
  logger->flush_on(spdlog::level::info);
  return logger;
}

std::shared_ptr<spdlog::logger> &Logger() noexcept {
  static std::shared_ptr<spdlog::logger> logger = [] {
    try {
      return CreateLogger();
    } catch (...) {
      return std::shared_ptr<spdlog::logger>{};
    }
  }();
  return logger;
}

void EmergencyWrite(const char *message, const char *error = nullptr) noexcept {
  // The logger must never break a transfer. This path is only used if spdlog
  // initialization or a sink fails, so keeping it dependency-free is useful.
  (void)std::fprintf(stderr, "[FLEXKV] ERROR [FlexKV-Native] %s%s%s%s\n",
                     message, error == nullptr ? "" : " error=\"",
                     error == nullptr ? "" : error,
                     error == nullptr ? "" : "\"");
  (void)std::fflush(stderr);
}

} // namespace

bool IsEnabled(Level level) noexcept {
  const auto &logger = Logger();
  if (logger != nullptr)
    return logger->should_log(ToSpdlogLevel(level));
  return level == Level::Error || level == Level::Critical;
}

void Logf(Level level, const char *file, int line, const char *format,
          ...) noexcept {
  if (format == nullptr)
    return;

  try {
    auto &logger = Logger();
    if (logger != nullptr && !logger->should_log(ToSpdlogLevel(level)))
      return;
    if (logger == nullptr && level != Level::Error && level != Level::Critical)
      return;

    char buffer[2048];
    va_list args;
    va_start(args, format);
    const int written = std::vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    if (written < 0)
      return;

    if (logger != nullptr) {
      logger->log(spdlog::source_loc{file, line, ""}, ToSpdlogLevel(level),
                  "{}", buffer);
    } else {
      EmergencyWrite(buffer);
    }
  } catch (const std::exception &error) {
    EmergencyWrite("operation=logging action=emit status=failed", error.what());
  } catch (...) {
    EmergencyWrite("operation=logging action=emit status=failed error=unknown");
  }
}

} // namespace flexkv::logging
