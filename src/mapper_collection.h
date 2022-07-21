#pragma once
#include "mapper.h"

namespace torchrec_mapper {

struct FutureCollection {
  FutureCollection(std::vector<Future> futures) : futures_(std::move(futures)) {}

  bool Wait() {
    bool result = true;
    for (size_t i = 0; i < futures_.size(); i++) {
      result = result && futures_[i].Wait();
    }
    return result;
  }

  std::vector<Future> futures_;
};

struct MapperCollection {
  MapperCollection(int64_t num_embedding, int64_t num_mappers)
    : num_mappers_(num_mappers) {
    int64_t embedding_per_mapper = num_embedding / num_mappers;
    for (int64_t i = 0; i < num_mappers; i++) {
      int64_t start = i * embedding_per_mapper;
      int64_t num = i == num_mappers - 1 ? num_embedding - i * embedding_per_mapper : embedding_per_mapper;
      mappers_.emplace_back(new Mapper(num_mappers, i, start, num));
    }
  }

  FutureCollection Map(torch::Tensor global_ids, torch::Tensor cache_ids, int32_t timestamp) {
    TORCH_CHECK(global_ids.is_cpu() && global_ids.scalar_type() == c10::kLong &&
                global_ids.is_contiguous());
    TORCH_CHECK(cache_ids.is_cpu() && cache_ids.scalar_type() == c10::kLong &&
                cache_ids.is_contiguous());
    std::vector<Future> futures;
    for (int64_t i = 0; i < num_mappers_; i++) {
      Future future = mappers_[i]->Map(global_ids, cache_ids, timestamp);
      futures.emplace_back(std::move(future));
    }

    return FutureCollection(std::move(futures));
  }

  int64_t num_mappers_;
  std::vector<std::unique_ptr<Mapper>> mappers_;
};

}
