#pragma once

#include <torch/extension.h>

namespace flexkv {

int get_hash_size();

void hash_tensor(const torch::Tensor &tensor, const torch::Tensor &result);

} // namespace flexkv
