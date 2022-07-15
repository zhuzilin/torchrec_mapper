#pragma once

#include <torch/torch.h>
#include <memory>
#include "fixed_size_hashmap.h"

namespace torchrec_mapper {

struct FreeSlotRange {
  int64_t start_;
  int64_t end_;
  std::unique_ptr<FreeSlotRange> next_;
};

struct Mapper {
  Mapper(int64_t num_embedding) :
    num_embedding_(num_embedding),
    global_id2cache_id_(2 * num_embedding),
    timestamps_(new int32_t[num_embedding]),
    free_slot_list_(new FreeSlotRange{0, num_embedding, nullptr}) {
    TORCH_CHECK(num_embedding > 0);
  }

  bool Map(torch::Tensor global_ids, torch::Tensor cache_ids, int32_t timestamp) {
    TORCH_CHECK(global_ids.is_cpu() && global_ids.scalar_type() == c10::kLong &&
                global_ids.is_contiguous());
    TORCH_CHECK(cache_ids.is_cpu() && cache_ids.scalar_type() == c10::kLong &&
                cache_ids.is_contiguous());
    int64_t n = cache_ids.numel();
    int64_t *src = global_ids.template data_ptr<int64_t>();
    int64_t *dst = cache_ids.template data_ptr<int64_t>();
    std::transform(src, src + n, dst,
                  [this, timestamp] (int64_t global_id) -> int64_t {
                    if (free_slot_list_ == nullptr)
                      return -1;
                    int64_t new_cache_id = free_slot_list_->start_;
                    int64_t cache_id = global_id2cache_id_.find_or_set(global_id, new_cache_id);
                    if (cache_id != -1) {
                      timestamps_[cache_id] = timestamp;
                      return cache_id;
                    } else {
                        free_slot_list_->start_++;
                        if (free_slot_list_->start_ == free_slot_list_->end_) {
                          free_slot_list_.reset(free_slot_list_->next_.release());
                        }
                        timestamps_[new_cache_id] = timestamp;
                        return new_cache_id;
                    }
                  });
    if (free_slot_list_ == nullptr) {
      return false;
    }
    return true;
  }

  int64_t num_embedding_;
  FixedSizeHashMap global_id2cache_id_;
  std::unique_ptr<int32_t[]> timestamps_;
  std::unique_ptr<FreeSlotRange> free_slot_list_;
};

};
