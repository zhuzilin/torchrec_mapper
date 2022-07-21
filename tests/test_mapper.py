import torch
from torchrec_mapper_cpp import Mapper
import tqdm


class PythonMapper:
    def __init__(self, num_embedding, num_mapper):
        self._num_embedding = num_embedding
        self._num_mapper = num_mapper
        self.dict = [{} for _ in range(num_mapper)]
        embedding_per_mapper = num_embedding // num_mapper
        self._mapper_start = [
            i * embedding_per_mapper for i in range(num_mapper)
        ] + [num_embedding]
        self.overflow = False

    def map(self, global_ids: torch.Tensor):
        global_id_list = global_ids.tolist()
        cache_id_list = [0] * len(global_id_list)
        for i in range(len(global_id_list)):
            gid = global_id_list[i]
            mapper_id = gid % self._num_mapper
            if gid in self.dict[mapper_id]:
                cid = self.dict[mapper_id][gid]
            else:
                cid = len(self.dict[mapper_id]) + self._mapper_start[mapper_id]
                self.dict[mapper_id][gid] = cid
                if len(self.dict[mapper_id]) >= self._mapper_start[
                        mapper_id + 1] - self._mapper_start[mapper_id]:
                    self.overflow = True
            cache_id_list[i] = cid
        cache_ids = torch.tensor(cache_id_list, dtype=torch.long)
        return cache_ids


if __name__ == "__main__":
    num_embedding = 1024
    num_mapper = 2
    mapper = Mapper(num_embedding, num_mapper)
    python_mapper = PythonMapper(num_embedding, num_mapper)

    batch_size = 16

    for timestamp in range(100):
        global_ids = torch.randint(2048, 4096, (batch_size, ))
        cache_ids = torch.empty_like(global_ids)
        future = mapper.map(global_ids, cache_ids, timestamp)
        succeed = future.wait()
        python_cache_ids = python_mapper.map(global_ids)
        if succeed:
            assert torch.all(python_cache_ids == cache_ids)
        else:
            assert python_mapper.overflow, f"python_mapper size: {len(python_mapper.dict)}"
            print(f"overflow on timestamp: {timestamp}")
            break
