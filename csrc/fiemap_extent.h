#pragma once

#include <vector>

namespace flexkv {

struct FileExtent {
    std::uint64_t logical;  // Logical offset on file in B
    std::uint64_t physical; // Physical offset on block device in B
    std::uint64_t length;   // Length in B
};

std::vector<FileExtent> get_fm_extents(int fd, uint32_t max_extents = 256);

} // namespace flexkv
