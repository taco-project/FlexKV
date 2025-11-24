#pragma once
#include <errno.h>
#include <torch/extension.h>
#include <vector>
#include <numeric>
#include <iostream>
#include <execinfo.h>

#include "cache_utils.h"

namespace flexkv {

class BlockInfo {
private:
  int refcnt;
  HashType hash;

public:
  BlockInfo() {
    refcnt = 0;
    hash = 0;
  }

  void reset() {
    refcnt = 0;
    hash = 0;
  }

  void set_hash(HashType hash) {
    this->hash = hash;
  }

  HashType get_hash() {
    return hash;
  }

  void set_ref(int ref) {
    assert(ref >= 0);
    assert(refcnt >= 0);

    refcnt = ref;
  }

  int get_ref() {
    return refcnt;
  }

  void inc_ref() {
    assert(refcnt >= 0);

    refcnt++;
  }

  void dec_ref() {
    assert(refcnt > 0);

    refcnt--;
  }
};

class CMemPool {
private:
  bool block_dedup;
  int num_total_blocks;
  int num_free;
  int free_ids_offset;

  std::vector<BlockInfo> *total_block_list;
  std::vector<int64_t> *free_blockid_list;
  std::unordered_map<HashType, int64_t> *hash_id_map;

public:
  CMemPool(int num_total_blocks, bool block_dedup) {
    assert(num_total_blocks > 0);

    this->block_dedup = block_dedup;
    this->num_total_blocks = num_total_blocks;
    this->num_free = num_total_blocks;
    this->free_ids_offset = 0;

    total_block_list = new std::vector<BlockInfo>(num_total_blocks);
    free_ids = new std::vector<int64_t>(num_total_blocks);
    std::iota(free_ids->begin(), free_ids->end(), 0);
    hash_id_map = new std::unordered_map<int64_t, int64_t>();
  }

  ~CMemPool() {
    delete hash_id_map;
    delete free_ids;
    delete total_block_list;
  };

  void reset() {
    this->num_free = this->num_total_blocks;
    this->free_ids_offset = 0;

    for (auto& it = total_block_list->begin(); it != total_block_list->end(); it++) {
      it->set_hash(0);
      it->set_ref(0);
    }

    std::iota(free_ids->begin(), free_ids->end(), 0);
    hash_id_map->clear();
  }

  void allocate_blocks(int num, torch::Tensor &block_hash_list, torch::Tensor &block_id_list, torch::Tensor &block_ref_list) {
    auto *block_hash_list_ptr = block_hash_list.data_ptr<int64_t>();
    auto *block_id_list_ptr = block_id_list.data_ptr<int64_t>();
    auto *block_ref_list_ptr = block_ref_list.data_ptr<int64_t>();
    int i;

    assert(block_hash_list.dim() == 1);
    assert(block_hash_list.size() != num);

    assert(block_id_list.dim() == 1);
    assert(block_id_list.size() != num);

    assert(block_ref_list.dim() == 1);
    assert(block_ref_list.size() != num);

    if (num <= 0) {
      throw std::runtime_error("num must be greater than 0, but got " + std::to_string(num));
    }

    if (num > num_free) {
      throw std::runtime_error("Not enough free blocks, required=" + std::to_string(num) + ", available=" + std::to_string(num_free));
    }

    if (num > (free_ids->size() - free_ids_offset)) {
      update_free_ids();
    }

    for (i = 0; i < num; i++) {
      if (block_dedup) {
        auto iter = hash_id_map->find(block_hash_list_ptr[i]);
        if (iter != hash_id_map->end()) {
          auto& block = total_block_list[iter.second()];

	  assert(block.get_hash() == block_hash_list_ptr[i]);
	  block.inc_ref();

	  block_id_list_ptr[i] = iter.second();
	  block_ref_list_ptr[i] = block.get_ref();
          num_free--;
	  continue;
	}
      }

      auto& block = total_block_list[free_ids[free_ids_offset]];

      block.set_ref(1);
      block.set_hash(block_hash_list_ptr[i]);
      block_id_list_ptr[i] = free_ids[free_ids_offset];
      block_ref_list_ptr[i] = 1;
      hash_id_map->insert(block_hash_list_ptr[i], free_ids->at(free_ids_offset));
      free_ids_offset++;
      num_free--;
    }
  }

  void recycle_blocks(torch::Tensor block_ids) {
    int block_ids_len = block_ids.size(0);
    int64_t *block_ids_ptr = block_ids.data_ptr<int64_t>();
    int i;

    if (block_ids.dim() != 1) {
      throw std::runtime_error("block_ids must be a 1D tensor of int64\n");
    }

    for (i = 0; i < block_ids_len; i++) {
      auto& block = total_block_list->at(block_ids_ptr[i]);

      if (block.get_ref() <= 1) {
        throw std::runtime_error("Cannot recycle free block_ids repeatedly: " + std::to_string(block_ids_ptr[i]));
      }

      block.dec_ref();
      if (block.get_ref() == 0) {
	if (block_dedup) {
          hash_id_map->erase(block.get_hash());
	}
	block.set_hash(0);
	num_free++;
      }
    }
  }

  void update_free_ids() {
    free_ids->clear();

    for (auto& it = total_block_list->begin(); it != total_block_list->end(); it++) {
      if (it->get_ref() == 0) {
        free_ids->push_back(std::distance(total_block_list->begin(), it));
      }
    }
      
    free_ids_offset = 0; 
  }

  int num_free_blocks() {
    return num_free;
  }

  int num_used_blocks() {
    return num_total_blocks - num_free;
  }
};

} // namespace flexkv
