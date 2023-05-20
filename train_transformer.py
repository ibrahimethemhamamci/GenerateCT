from phenaki_pytorch import CViViT, MaskGit, Phenaki
from phenaki_pytorch.videotextdataset import VideoTextDataset
from phenaki_pytorch.train_phenaki import PhenakiTrainer
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import os

def cycle(dl):
    while True:
        for data in dl:
            yield data



def train():
    # set up distributed training

    cvivit = CViViT(
        dim = 512,
        codebook_size = 8192,
        image_size = 128,
        patch_size = 16,
        temporal_patch_size = 2,
        spatial_depth = 4,
        temporal_depth = 4,
        dim_head = 32,
        heads = 8
    )
    
   


    

    # Load the pre-trained weights

    pretrained_cvivit_path = 'scratch/128_16_200f_batch4_8gpus_triple_continue2/vae.3000.pt'
    cvivit.load(pretrained_cvivit_path)

    maskgit = MaskGit(
        num_tokens=8192,
        max_seq_len=10000,
        dim=512,
        dim_context=768,
        depth=6,
    )
    
    

    phenaki_model = Phenaki(
        cvivit=cvivit,
        maskgit=maskgit
    )
    batch_size=1
    phenaki_model.load('pretrained_transformer/transformer.pretrained.pt')


    # initialize DDP
    trainer = PhenakiTrainer(
        phenaki_model,
        num_train_steps=100000000,
        batch_size=1,
        pretrained_cvivit_path='pretrained_ctvit/vae.pretrained.pt',
        results_folder="transformer"
    )


    trainer.train()

if __name__ == '__main__':
    # set up multiprocessing
    train()
