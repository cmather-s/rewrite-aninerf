import torch.utils.data as data
from dataset import Dataset
import transform
import sampler
import numpy as np 
import time

def worker_init_fn(worker_id):
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))

def make_dataloader():

    batch_size = 1
    shuffle = True
    drop_last = False

    dataset = Dataset('data/zju_mocap/CoreView_313')
    transform = transform.make_transform()
    batch_sampler = sampler.ImageSizeBatchSampler()

    data_loader = data.DataLoader(dataset,batch_sampler=batch_sampler,num_workers=0,worker_init_fn=worker_init_fn)

    return data_loader
