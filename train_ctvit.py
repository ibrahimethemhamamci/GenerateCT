import torch
from transformer_maskgit import CTiViT, CTViTTrainer

cvivit = CTViT(
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

trainer = CTViTTrainer(
    cvivit,
    folder = 'example_data/ctvit-transformer',
    batch_size = 4,
    results_folder="ctvit",
    grad_accum_every = 1,
    train_on_images = False,  # you can train on images first, before fine tuning on video, for sample efficiency
    use_ema = False,          # recommended to be turned on (keeps exponential moving averaged cvivit) unless if you don't have enough resources
    num_train_steps = 2000000,
    num_frames=2
)

trainer.train()               # reconstructions and checkpoints will be saved periodically to ./results
