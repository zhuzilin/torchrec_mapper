import torch
from torchrec_mapper_cpp import Mapper
import tqdm


class PythonMapper:
    def __init__(self, num_embedding):
        self._num_embedding = num_embedding
        self.dict = {}

    def map(self, global_ids: torch.Tensor):
        global_id_list = global_ids.tolist()
        cache_id_list = [0] * len(global_id_list)
        for i in range(len(global_id_list)):
            gid = global_id_list[i]
            if gid in self.dict:
                cid = self.dict[gid]
            else:
                cid = len(self.dict)
                self.dict[gid] = cid
            cache_id_list[i] = cid
        cache_ids = torch.tensor(cache_id_list, dtype=torch.long)
        return cache_ids


if __name__ == "__main__":
    num_embedding = 1024
    mapper = Mapper(num_embedding)
    python_mapper = PythonMapper(num_embedding)

    batch_size = 16
    cache_ids = torch.empty((batch_size, ), dtype=torch.long)

    for timestamp in range(100):
        global_ids = torch.randint(100000000, 200000000, (batch_size, ))
        succeed = mapper.map(global_ids, cache_ids, timestamp)
        python_cache_ids = python_mapper.map(global_ids)

        if succeed:
            assert torch.all(python_cache_ids == cache_ids)
        else:
            assert len(
                python_mapper.dict
            ) >= num_embedding, f"python_mapper size: {len(python_mapper.dict)}"
