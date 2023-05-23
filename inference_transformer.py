from transformer_maskgit import CTViT, MaskGit, MaskGITTransformer
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import os
from torch.utils.data import DataLoader
from transformer_maskgit.data import tensor_to_nifti
from pathlib import Path
import glob
import pandas as pd
import random



def cycle(dl):
    while True:
        for data in dl:
            yield data

def scan_folder(directory):
    # Use a set to automatically eliminate duplicates
    txt_content = set()

    # Use glob to recursively find all .txt files
    for filename in glob.glob(directory + '/**/*.txt', recursive=True):
        with open(filename, 'r') as f:
            content = f.readline().strip()
            txt_content.add(content)

    return list(txt_content)



def infer():
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

    cvivit.eval()

    # Load the pre-trained weights

    pretrained_cvivit_path = 'pretrained_models/ctvit_pretrained.pt'
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
    ).cuda()
    batch_size=1
    transformer_model.load('pretrained_models/transformer_pretrained.pt')
    phenaki_model.eval()

    xlsx_file = 'example_data/text_prompts.xlsx'
    labels_data = pd.read_excel(xlsx_file)

    # Create a dictionary to map table names to text prompts for given labels
    texts_dict = dict(zip(labels_data['Names'], labels_data['Text_prompts']))
    #for i in range(len(texts)):
    for i, (input_name, text) in enumerate(texts_dict.items()):
        for k in range(1):
            out=phenaki_model.sample(texts =text, num_frames = 201, cond_scale = 5.)
            path_test=Path("phenaki_inference")
            sampled_videos_path = path_test / f'samples.{input_name}_{str(i)}'
            (sampled_videos_path).mkdir(parents = True, exist_ok = True)
            for tensor in out.unbind(dim = 0):
                tensor_to_nifti(tensor, str(sampled_videos_path / f'{input_name}_{k}.nii.gz'))
                filename = str(sampled_videos_path / f'{input_name}_{k}.txt')
                with open(filename, "w", encoding="utf-8") as file:
                    file.write(text)

if __name__ == '__main__':
    # set up multiprocessing
    infer()
