#pragma once

namespace flexkv::logging {

enum class Level {
  Debug,
  Info,
  Warning,
  Error,
  Critical,
};

bool IsEnabled(Level level) noexcept;
void Logf(Level level, const char *file, int line, const char *format,
          ...) noexcept;

} // namespace flexkv::logging

#define FLEXKV_LOG_IMPL(level, ...)                                          \
  do {                                                                       \
    if (::flexkv::logging::IsEnabled(::flexkv::logging::Level::level)) {     \
      ::flexkv::logging::Logf(::flexkv::logging::Level::level, __FILE__,     \
                              __LINE__, __VA_ARGS__);                         \
    }                                                                        \
  } while (false)

#define FLEXKV_LOG_DEBUG(...) FLEXKV_LOG_IMPL(Debug, __VA_ARGS__)
#define FLEXKV_LOG_INFO(...) FLEXKV_LOG_IMPL(Info, __VA_ARGS__)
#define FLEXKV_LOG_WARNING(...) FLEXKV_LOG_IMPL(Warning, __VA_ARGS__)
#define FLEXKV_LOG_ERROR(...) FLEXKV_LOG_IMPL(Error, __VA_ARGS__)
#define FLEXKV_LOG_CRITICAL(...) FLEXKV_LOG_IMPL(Critical, __VA_ARGS__)
