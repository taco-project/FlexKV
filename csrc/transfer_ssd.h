#include <errno.h>
#include <fcntl.h>
#include <torch/extension.h>
#include <unistd.h>
#include <vector>

#include <mutex>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

void transfer_kv_blocks_ssd_naive(
    const std::string &filename, const torch::Tensor &cpu_layer_ptrs_tensor,
    const torch::Tensor &ssd_block_ids, const torch::Tensor &cpu_block_ids,
    int64_t cpu_kv_stride_in_bytes, int64_t ssd_layer_stride_in_bytes,
    int64_t ssd_block_stride_in_bytes, int64_t ssd_kv_stride_in_bytes,
    int64_t block_size_in_bytes, bool is_read, bool verbose = false) {

  const int64_t *layer_ptrs = cpu_layer_ptrs_tensor.data_ptr<int64_t>();
  const int64_t *ssd_block_id_ptr = ssd_block_ids.data_ptr<int64_t>();
  const int64_t *cpu_block_id_ptr = cpu_block_ids.data_ptr<int64_t>();

  const int num_layers = cpu_layer_ptrs_tensor.size(0);
  const int num_transfers = ssd_block_ids.size(0);

  // open file
  int flags = is_read ? O_RDONLY : (O_RDWR | O_CREAT);
  int o_direct_flag = block_size_in_bytes % 4096 == 0 ? O_DIRECT : 0;
  int fd = open(filename.c_str(), flags | o_direct_flag, 0644);
  if (fd < 0) {
    throw std::runtime_error("Failed to open file: " +
                             std::string(strerror(errno)));
  }

  for (int i = 0; i < num_layers; i++) {
    void *layer_ptr = reinterpret_cast<void *>(layer_ptrs[i]);

    void *k_view = layer_ptr;
    void *v_view = layer_ptr + cpu_kv_stride_in_bytes;

    for (int j = 0; j < num_transfers; j++) {
      int64_t ssd_block_id = ssd_block_id_ptr[j];
      int64_t cpu_block_id = cpu_block_id_ptr[j];

      int64_t ssd_base_offset = ssd_layer_stride_in_bytes * i +
                                ssd_block_stride_in_bytes * ssd_block_id;

      // process K
      void *k_ptr = k_view + cpu_block_id * block_size_in_bytes;

      if (is_read) {
        ssize_t bytes_read =
            pread(fd, k_ptr, block_size_in_bytes, ssd_base_offset);
        if (bytes_read != block_size_in_bytes) {
          close(fd);
          throw std::runtime_error("Failed to read K block: " +
                                   std::string(strerror(errno)));
        }
      } else {
        ssize_t bytes_written =
            pwrite(fd, k_ptr, block_size_in_bytes, ssd_base_offset);
        if (bytes_written != block_size_in_bytes) {
          close(fd);
          throw std::runtime_error("Failed to write K block: " +
                                   std::string(strerror(errno)));
        }
      }

      // process V
      void *v_ptr = v_view + cpu_block_id * block_size_in_bytes;
      int64_t v_offset = ssd_base_offset + ssd_kv_stride_in_bytes;

      if (is_read) {
        ssize_t bytes_read = pread(fd, v_ptr, block_size_in_bytes, v_offset);
        if (bytes_read != block_size_in_bytes) {
          close(fd);
          throw std::runtime_error("Failed to read V block: " +
                                   std::string(strerror(errno)));
        }
      } else {
        ssize_t bytes_written =
            pwrite(fd, v_ptr, block_size_in_bytes, v_offset);
        if (bytes_written != block_size_in_bytes) {
          close(fd);
          throw std::runtime_error("Failed to write V block: " +
                                   std::string(strerror(errno)));
        }
      }

      if (verbose) {
        std::cerr << "Layer " << i << " Block " << j
                  << " Operation: " << (is_read ? "Read" : "Write")
                  << " SSD Block ID: " << ssd_block_id
                  << " CPU Block ID: " << cpu_block_id << std::endl;
      }
    }
  }

  if (!is_read) {
    fsync(fd);
  }
  close(fd);
}
// NOTE that we may also use other techniques such as
// AIO, O_DIRECT, and etc to improve the performance
void transfer_kv_blocks_ssd_mmap_multi_thread(
    const std::string &filename, const torch::Tensor &cpu_layer_ptrs_tensor,
    const torch::Tensor &ssd_block_ids, const torch::Tensor &cpu_block_ids,
    int64_t cpu_kv_stride_in_bytes, int64_t ssd_layer_stride_in_bytes,
    int64_t ssd_block_stride_in_bytes, int64_t ssd_kv_stride_in_bytes,
    int64_t block_size_in_bytes, bool is_read, bool verbose = false) {

  const int64_t *layer_ptrs = cpu_layer_ptrs_tensor.data_ptr<int64_t>();
  const int64_t *ssd_block_id_ptr = ssd_block_ids.data_ptr<int64_t>();
  const int64_t *cpu_block_id_ptr = cpu_block_ids.data_ptr<int64_t>();

  const int num_layers = cpu_layer_ptrs_tensor.size(0);
  const int num_transfers = ssd_block_ids.size(0);

  int64_t file_size = ssd_layer_stride_in_bytes * num_layers;

  int o_direct_flag = block_size_in_bytes % 4096 == 0 ? O_DIRECT : 0;

  if (is_read) {
    // here we use multiple threads to read.
    int fd = open(filename.c_str(), O_RDONLY | o_direct_flag);
    if (fd < 0) {
      throw std::runtime_error("Failed to open file for reading: " +
                               std::string(strerror(errno)));
    }

    // read hint
    posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL | POSIX_FADV_WILLNEED);

    // create multiple threads to handle different layers
    const int num_threads =
        std::min(8, num_layers); // limit the max number of threads
    std::vector<std::thread> threads;
    std::mutex error_mutex;
    std::string error_message;

    auto process_layers = [&](int start_layer, int end_layer) {
      try {
        // each thread opens a separate file descriptor
        int thread_fd = open(filename.c_str(), O_RDONLY | o_direct_flag);
        if (thread_fd < 0) {
          throw std::runtime_error("Thread failed to open file: " +
                                   std::string(strerror(errno)));
        }

        for (int i = start_layer; i < end_layer; i++) {
          void *layer_ptr = reinterpret_cast<void *>(layer_ptrs[i]);
          void *k_view = layer_ptr;
          void *v_view =
              static_cast<char *>(layer_ptr) + cpu_kv_stride_in_bytes;

          for (int j = 0; j < num_transfers; j++) {
            int64_t ssd_block_id = ssd_block_id_ptr[j];
            int64_t cpu_block_id = cpu_block_id_ptr[j];

            int64_t ssd_base_offset = ssd_layer_stride_in_bytes * i +
                                      ssd_block_stride_in_bytes * ssd_block_id;

            // read K block
            void *k_ptr = static_cast<char *>(k_view) +
                          cpu_block_id * block_size_in_bytes;
            ssize_t bytes_read =
                pread(thread_fd, k_ptr, block_size_in_bytes, ssd_base_offset);
            if (bytes_read != block_size_in_bytes) {
              throw std::runtime_error("Failed to read K block");
            }

            // read V block
            void *v_ptr = static_cast<char *>(v_view) +
                          cpu_block_id * block_size_in_bytes;
            int64_t v_offset = ssd_base_offset + ssd_kv_stride_in_bytes;
            bytes_read = pread(thread_fd, v_ptr, block_size_in_bytes, v_offset);
            if (bytes_read != block_size_in_bytes) {
              throw std::runtime_error("Failed to read V block");
            }

            if (verbose) {
              std::lock_guard<std::mutex> lock(error_mutex);
              std::cerr << "Thread " << std::this_thread::get_id()
                        << " read Layer " << i << " Block " << j << std::endl;
            }
          }
        }
        close(thread_fd);
      } catch (const std::exception &e) {
        std::lock_guard<std::mutex> lock(error_mutex);
        error_message = e.what();
      }
    };

    // assign layers to each thread
    int layers_per_thread = (num_layers + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; t++) {
      int start_layer = t * layers_per_thread;
      int end_layer = std::min(start_layer + layers_per_thread, num_layers);
      if (start_layer < end_layer) {
        threads.emplace_back(process_layers, start_layer, end_layer);
      }
    }

    // wait for all threads to finish
    for (auto &thread : threads) {
      thread.join();
    }

    close(fd);

    // check if any error occurs
    if (!error_message.empty()) {
      throw std::runtime_error("Error in read thread: " + error_message);
    }

  } else {
    // here we use mmap to write.
    int fd = open(filename.c_str(), O_RDWR | O_CREAT | o_direct_flag, 0644);
    if (fd < 0) {
      throw std::runtime_error("Failed to open file for writing: " +
                               std::string(strerror(errno)));
    }

    // resize the file
    if (ftruncate(fd, file_size) < 0) {
      close(fd);
      throw std::runtime_error("Failed to resize file");
    }

    // map the whole file
    void *mmap_ptr =
        mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mmap_ptr == MAP_FAILED) {
      close(fd);
      throw std::runtime_error("Failed to mmap file: " +
                               std::string(strerror(errno)));
    }

    // write hint, MADV_SEQUENTIAL means that the data is sequential
    // MADV_DONTNEED means that the data will not be needed soon
    madvise(mmap_ptr, file_size, MADV_SEQUENTIAL | MADV_DONTNEED);

    // create multiple threads to write
    const int num_threads = std::min(8, num_layers);
    std::vector<std::thread> threads;
    std::mutex error_mutex;
    std::string error_message;

    auto process_layers = [&](int start_layer, int end_layer) {
      try {
        for (int i = start_layer; i < end_layer; i++) {
          void *layer_ptr = reinterpret_cast<void *>(layer_ptrs[i]);
          void *k_view = layer_ptr;
          void *v_view =
              static_cast<char *>(layer_ptr) + cpu_kv_stride_in_bytes;

          for (int j = 0; j < num_transfers; j++) {
            int64_t ssd_block_id = ssd_block_id_ptr[j];
            int64_t cpu_block_id = cpu_block_id_ptr[j];

            int64_t ssd_base_offset = ssd_layer_stride_in_bytes * i +
                                      ssd_block_stride_in_bytes * ssd_block_id;

            // write K block
            void *k_ptr = static_cast<char *>(k_view) +
                          cpu_block_id * block_size_in_bytes;
            void *mmap_k_ptr = static_cast<char *>(mmap_ptr) + ssd_base_offset;
            memcpy(mmap_k_ptr, k_ptr, block_size_in_bytes);

            // write V block
            void *v_ptr = static_cast<char *>(v_view) +
                          cpu_block_id * block_size_in_bytes;
            void *mmap_v_ptr = static_cast<char *>(mmap_ptr) + ssd_base_offset +
                               ssd_kv_stride_in_bytes;
            memcpy(mmap_v_ptr, v_ptr, block_size_in_bytes);

            if (verbose) {
              std::lock_guard<std::mutex> lock(error_mutex);
              std::cerr << "Thread " << std::this_thread::get_id()
                        << " wrote Layer " << i << " Block " << j << std::endl;
            }
          }
        }
      } catch (const std::exception &e) {
        std::lock_guard<std::mutex> lock(error_mutex);
        error_message = e.what();
      }
    };

    // assign layers to each thread
    int layers_per_thread = (num_layers + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; t++) {
      int start_layer = t * layers_per_thread;
      int end_layer = std::min(start_layer + layers_per_thread, num_layers);
      if (start_layer < end_layer) {
        threads.emplace_back(process_layers, start_layer, end_layer);
      }
    }

    // wait for all threads to finish
    for (auto &thread : threads) {
      thread.join();
    }

    // check if any error occurs
    if (!error_message.empty()) {
      munmap(mmap_ptr, file_size);
      close(fd);
      throw std::runtime_error("Error in write thread: " + error_message);
    }

    // sync and clean up
    msync(mmap_ptr, file_size, MS_SYNC);
    munmap(mmap_ptr, file_size);
    close(fd);
  }
}
