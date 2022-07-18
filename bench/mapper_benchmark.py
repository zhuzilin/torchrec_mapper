import torch
from torchrec_mapper_cpp import Mapper
from tqdm import tqdm

if __name__ == "__main__":
    num_embedding = int(3e8)
    mapper = Mapper(num_embedding)

    hot_percentage = 0.5
    total_size = 1024
    hot_size = int(hot_percentage * total_size)
    batch_size = 1024
    cache_ids = torch.empty((batch_size, total_size), dtype=torch.long)
    global_ids = torch.empty((batch_size, total_size), dtype=torch.long)

    for timestamp in tqdm(range(200)):
        global_ids[:, :hot_size].random_(0, int(1e6))
        global_ids[:, hot_size:].random_(0, int(1e10))
        future = mapper.map(global_ids, cache_ids, timestamp)
        future.wait()
