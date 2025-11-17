#include "fiemap_extent.h"

#include <linux/fiemap.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <cstring>
#if __has_include(<format>)
#include <format>
#endif

namespace flexkv {

std::vector<FileExtent> get_fm_extents(int fd, uint32_t max_extents) {
    if (fd < 0) [[unlikely]] {
        throw std::invalid_argument("get_fm_extents: fd < 0");
    }

    size_t fm_size = sizeof(struct fiemap) + max_extents * sizeof(struct fiemap_extent);
    auto* fm = static_cast<struct fiemap*>(std::calloc(1, fm_size));
    if (!fm) {
        throw std::bad_alloc();
    }

    fm->fm_start = 0;
    fm->fm_length = ~0ULL;           // map whole file
    fm->fm_flags = FIEMAP_FLAG_SYNC; // sync before mapping
    fm->fm_extent_count = max_extents;

    if (ioctl(fd, FS_IOC_FIEMAP, fm) < 0) [[unlikely]] {
        int err = errno;
        std::free(fm);
#if __has_include(<format>)
        std::string msg = std::format("FS_IOC_FIEMAP failed: {}", std::strerror(err));
#else
        std::string msg = "FS_IOC_FIEMAP failed: " + std::string(std::strerror(err));
#endif
        throw std::runtime_error(msg);
    }

    std::vector<FileExtent> out;
    out.reserve(fm->fm_mapped_extents);
    for (unsigned int i = 0; i < fm->fm_mapped_extents; i++) {
        const auto &fe = fm->fm_extents[i];
        if (fe.fe_length == 0) {
            continue;
        }
        out.push_back(FileExtent{fe.fe_logical, fe.fe_physical, fe.fe_length});
    }

    std::free(fm);
    return out;
}

}
