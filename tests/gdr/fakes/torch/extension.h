#pragma once
#include <cstdint>
#include <vector>
#include <sstream>
#include <stdexcept>
namespace fake_torch {
template <typename... Args>
[[noreturn]] void check_failed(const Args &...args) {
  std::ostringstream message;
  (message << ... << args);
  throw std::runtime_error(message.str());
}
}
#define TORCH_CHECK(condition, ...) \
  do { if (!(condition)) fake_torch::check_failed(__VA_ARGS__); } while (false)
namespace torch {
constexpr int kCPU=0;
class Tensor {
  std::vector<int32_t> data_;
 public:
  Tensor()=default;
  explicit Tensor(const std::vector<int32_t>& v):data_(v) {}
  bool defined() const {return !data_.empty();}
  size_t numel() const {return data_.size();}
  Tensor to(int) const {return *this;}
  Tensor contiguous() const {return *this;}
  template<class T> T *data_ptr() const {return reinterpret_cast<T*>(const_cast<int32_t*>(data_.data()));}
};
}
