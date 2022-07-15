import torch
from torchrec_mapper_cpp import Mapper
from tqdm import tqdm

if __name__ == "__main__":
    num_embedding = int(3e8)
    mapper = Mapper(num_embedding)

    batch_shape = (1024, 1024)
    cache_ids = torch.empty(batch_shape, dtype=torch.long)

    for timestamp in tqdm(range(1000)):
        global_ids = torch.randint(0, int(1e10), batch_shape)
        succeed = mapper.map(global_ids, cache_ids, timestamp)
