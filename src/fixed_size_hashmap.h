#pragma once

namespace torchrec_mapper {

// A fixed size open address hash map.
// Note that it's user's responsibility to make sure the hash table
// is not full.
class FixedSizeHashMap {
public:
  FixedSizeHashMap(int64_t capacity_) :
    capacity_(capacity_),
    buckets_(new int64_t[2 * capacity_]) {
    memset(buckets_.get(), -1, 2 * capacity_ * sizeof(int64_t));
  }

  int64_t find_or_set(int64_t key, int64_t val) {
    int64_t hash = (key % capacity_) * 2;
    do {
      if (buckets_[hash] == key) {
        return buckets_[hash + 1];
      }
      if (buckets_[hash] == -1) {
        buckets_[hash + 1] = val;
        return -1;
      }
      hash = (hash + 2) % capacity_;
    } while (true);
  }

private:
  const int64_t capacity_;
  int64_t size_ = 0;
  std::unique_ptr<int64_t[]> buckets_;
};

}
