#pragma once

#include <memory>
#include <c10/util/flat_hash_map.h>
#include <torch/torch.h>

namespace torchrec_mapper {

struct Request {
  torch::Tensor global_ids_;
  torch::Tensor cache_ids_;
  int64_t timestamp_;
};

struct Future {
  template<typename WaitFunction>
  Future(WaitFunction func) : wait_func_(std::move(func)) {}

  bool Wait() { return wait_func_(); }

  std::function<bool()> wait_func_;
};

struct Mapper {
  Mapper(int64_t num_embedding) :
    num_embedding_(num_embedding),
    global_id2cache_id_(2 * num_embedding),
    timestamps_(new int32_t[num_embedding]),
    new_cache_id_(0),
    shutdown_(false) {
    TORCH_CHECK(num_embedding > 0);
    work_thread_ = std::thread([this] {
      while (!shutdown_) {
        int64_t n, *src, *dst;
        int32_t timestamp;
        {
          std::unique_lock lock(req_mu_);
          req_cv_.wait(lock, [this] { return reqs_.size() > 0 || shutdown_; });
          if (shutdown_)
            break;
          Request req = reqs_.front();
          reqs_.pop();
          n = req.cache_ids_.numel();
          src = req.global_ids_.template data_ptr<int64_t>();
          dst = req.cache_ids_.template data_ptr<int64_t>();
          timestamp = req.timestamp_;
        }
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
        {
          std::unique_lock lock(rep_mu_);
          reps_.push(new_cache_id_ != num_embedding_);
        }
        rep_cv_.notify_one();
      }
    });
  }

  ~Mapper() {
    shutdown_ = true;
    req_cv_.notify_one();
    work_thread_.join();
  }

  Future Map(torch::Tensor global_ids, torch::Tensor cache_ids, int32_t timestamp) {
    TORCH_CHECK(global_ids.is_cpu() && global_ids.scalar_type() == c10::kLong &&
                global_ids.is_contiguous());
    TORCH_CHECK(cache_ids.is_cpu() && cache_ids.scalar_type() == c10::kLong &&
                cache_ids.is_contiguous());
    {
      std::unique_lock lock(req_mu_);
      reqs_.push(Request{global_ids, cache_ids, timestamp});
    }
    req_cv_.notify_one();


    return Future([this] {
      std::unique_lock lock(rep_mu_);
      rep_cv_.wait(lock, [this] { return reps_.size() > 0; });
      bool rep = reps_.front();
      reps_.pop();
      return rep;
    });
  }

  int64_t num_embedding_;
  ska::flat_hash_map<int64_t, int64_t> global_id2cache_id_;
  std::unique_ptr<int32_t[]> timestamps_;
  int64_t new_cache_id_;

  std::thread work_thread_;

  std::mutex req_mu_;
  std::condition_variable req_cv_;
  std::queue<Request> reqs_;

  std::mutex rep_mu_;
  std::condition_variable rep_cv_;
  std::queue<bool> reps_;

  bool shutdown_;
};

}
