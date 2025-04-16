#include <chrono>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <string>
#include <vector>

#include <torch/extension.h>

namespace flexkv {

int get_hash_size() { return SHA256_DIGEST_LENGTH; }

void hash_tensor(const torch::Tensor &tensor, const torch::Tensor &result) {
  assert(tensor.ndim() == 1);
  EVP_MD_CTX *ctx = EVP_MD_CTX_new();
  EVP_DigestInit_ex(ctx, EVP_sha256(), NULL);
  EVP_DigestUpdate(ctx, tensor.data_ptr(), tensor.numel() * sizeof(int));
  EVP_DigestFinal_ex(ctx, reinterpret_cast<unsigned char *>(result.data_ptr()),
                     NULL);
  EVP_MD_CTX_free(ctx);
}

} // namespace flexkv
