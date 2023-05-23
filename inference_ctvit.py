import torch
from transformer_maskgit import CTViT 
from transformer_maskgit.ctvit_inference import CTVIT_inf

ctvit = CTViT(
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
ctvit.load('pretrained_models/ctvit_pretrained.pt')

vit_infer = CTVIT_inf(
    ctvit,
    folder = 'example_data_valid_ctvit',
    batch_size = 1,
    results_folder="ctvit_inference",
    grad_accum_every = 1,
    train_on_images = False,  # you can train on images first, before fine tuning on video, for sample efficiency
    use_ema = False,          # recommended to be turned on (keeps exponential moving averaged ctvit) unless if you don't have enough resources
    num_train_steps = 1,
    num_frames=2
)

vit_infer.infer()               # reconstructions and checkpoints will be saved periodically to ./results

