#pragma once

#include <memory>
#include <c10/util/flat_hash_map.h>
#include <torch/torch.h>

namespace torchrec_mapper {

struct Mapper {
  Mapper(int64_t num_embedding) :
    num_embedding_(num_embedding),
    global_id2cache_id_(2 * num_embedding),
    timestamps_(new int32_t[num_embedding]),
    new_cache_id_(0) {
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
                    if (new_cache_id_ == num_embedding_)
                      return -1;
                    int64_t cache_id;
                    auto found = global_id2cache_id_.find(global_id);
                    if (found != global_id2cache_id_.end()) {
                      cache_id = found->second;
                    } else {
                      cache_id = new_cache_id_;
                      global_id2cache_id_.emplace(global_id, cache_id);
                      new_cache_id_++;
                    }
                    timestamps_[cache_id] = timestamp;
                    return cache_id;
                  });
    if (new_cache_id_ == num_embedding_) {
      return false;
    }
    return true;
  }

  int64_t num_embedding_;
  ska::flat_hash_map<int64_t, int64_t> global_id2cache_id_;
  std::unique_ptr<int32_t[]> timestamps_;
  int64_t new_cache_id_;
};

}
