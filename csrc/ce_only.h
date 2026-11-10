#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace flexkv {

inline bool ce_only_mode() {
  const char *raw = std::getenv("FLEXKV_ENABLE_CE_ONLY");
  if (!raw)
    return false;
  std::string value(raw);
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return false;
  value = value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (value == "1" || value == "true" || value == "yes" || value == "on")
    return true;
  if (value == "0" || value == "false" || value == "no" || value == "off")
    return false;
  throw std::invalid_argument("Invalid FLEXKV_ENABLE_CE_ONLY value");
}

} // namespace flexkv
