import os, shutil
from omegaconf import OmegaConf
import argparse
from datetime import datetime
import time
import torch
import numpy as np
from PIL import Image
import torch
from einops import rearrange
from scipy.ndimage import zoom
from super_resolution import Unet, ElucidatedSuperres, SuperResolutionTrainer, Superres, NullUnet
from transformer_maskgit.videotextdatasettransformersuperres import VideoTextDataset
import nibabel as nib
import numpy
import torch.nn.functional as F
from torch.utils.data import DataLoader


def delay2str(t):
    t = int(t)
    secs = t%60
    mins = (t//60)%60
    hours = (t//3600)%24
    days = t//86400
    string = f"{secs}s"
    if mins:
        string = f"{mins}m {string}"
    if hours:
        string = f"{hours}h {string}"
    if days:
        string = f"{days}d {string}"
    return string

def create_save_folder(config_path, save_folder):
    os.makedirs(save_folder, exist_ok = True)
    shutil.copy(config_path, os.path.join(save_folder, "config.yaml"))
    os.makedirs(os.path.join(save_folder, "images"), exist_ok = True)


def one_line_log(config, cur_step, loss, batch_per_epoch, start_time, validation=False):
    s_step = f'Step: {cur_step:<6}'
    s_loss = f'Loss: {loss:<6.4f}' if not validation else f'Val loss: {loss:<6.4f}'
    s_epoch = f'Epoch: {(cur_step//batch_per_epoch):<4.0f}'
    s_mvid = f'Mimg: {(cur_step*config.dataloader.params.batch_size/1e6):<6.4f}'
    s_delay = f'Elapsed time: {delay2str(time.time() - start_time):<10}'
    print(f'{s_step} | {s_loss} {s_epoch} {s_mvid} | {s_delay}', end='\r') # type: ignore


def cycle(dl):
    while True:
        for data in dl:
            yield data

def get_exp_name(args):
    exp_name = args.config.split("/")[-1].split(".")[0] # get config file name
    exp_name = f"{exp_name}_stage{args.stage}"
    if args.uname != "":
        exp_name = f"{exp_name}_{args.uname}"
    return exp_name

def update_config_with_arg(args, config):
    if args.bs != -1:
        config.dataloader.params.batch_size = args.bs
        config.dataloader.params.num_workers = min(args.bs, os.cpu_count())
        print(config.dataloader.params.num_workers)
        config.checkpoint.batch_size = min(args.bs, config.checkpoint.batch_size)

    if args.lr != -1:
        config.trainer.lr = args.lr

    if args.steps != -1:
        if config.superres.get("elucidated", True) == True:
            config.superres.num_sample_steps = args.steps
        else:
            config.superres.timesteps = args.steps

    return config

if __name__ == "__main__":

    torch.hub.set_dir(".cache")

    # Get config and args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")
    parser.add_argument("--resume", type=str, default="auto")
    parser.add_argument("--stage", type=int, default="1")
    parser.add_argument("--bs", type=int, default="-1")
    parser.add_argument("--lr", type=float, default="-1")
    parser.add_argument("--steps", type=int, default=-1, help="diffusion steps")
    parser.add_argument("--uname", type=str, default="", help="unique name for experiment")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, vars(args))

    # Define experiment name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = get_exp_name(args)

    # Overwrite config values with args
    config = update_config_with_arg(args, config)

    # Create models and inference
    unet1 = NullUnet()
    unet2=[Unet(**v, lowres_cond=(i>0)) for i, v in enumerate(config.unets.values())]

    superres_klass = ElucidatedSuperres if config.superres.get('elucidated', False) else Superres
    superres = superres_klass(
        unets = (unet1,unet2[0]),
        **OmegaConf.to_container(config.superres.params), # type: ignore
    )

    infer = SuperResolutionTrainer(
        superres = superres,
        **config.trainer.params,
    ).to(device)
    train_ds=VideoTextDataset(data_folder='example_data/superres/ctvit_outputs/', xlsx_file='example_data/data_reports.xlsx', num_frames=2)


    dl = DataLoader(train_ds)
    dl=cycle(dl)
    batch_per_epoch = (len(train_ds) // config.dataloader.params.batch_size)+1

    save_folder = os.path.join(config.checkpoint.path, exp_name)
    if infer.is_main:
    # Create save folder and resume logic
        if args.resume not in ['auto', 'overwrite']:
            raise ValueError("Got wrong resume value: ", args.resume)

    # Create save folder if it doesn't exist and copy config file
        create_save_folder(args.config, save_folder)

    infer.accelerator.wait_for_everyone()

    # Resume training if requested and possible
    weight_path = os.path.join("pretrained_models","superres_pretrained.pt")
    infer.accelerator.print(f"Inference from {weight_path}")
    additional_data = infer.load(weight_path)
    start_time = time.time() - additional_data["time_elapsed"] # type: ignore

    # Save reference videos and get test embeddings
    if infer.is_main:
        sample_kwargs = {}
        sample_kwargs["start_at_unet_number"] = config.stage
        sample_kwargs["stop_at_unet_number"] = config.stage

    infer.accelerator.print("Starting inference loop...")
    infer.accelerator.wait_for_everyone()

    cur_step = 0
    for i in range(len(train_ds)): # let slurm handle the stopping
        if True:
            infer.accelerator.wait_for_everyone()
            infer.accelerator.print()
            infer.accelerator.print(f'Saving videos (it. {cur_step})')

            if infer.is_main:
                images_ref_input, texts_ref, path_name= next(iter(dl))
                images_ref_input=images_ref_input[0]
                texts_ref=texts_ref[0]
                images_ref_input=images_ref_input.permute(1, 0, 2,3)
                image_ref_input_shape = images_ref_input[0].shape
                texts_ref=[texts_ref]
                sample_kwargs["texts"] = texts_ref

                with torch.no_grad():
                    image_list=[]
                    torch.cuda.empty_cache()
                    for k in range(images_ref_input.shape[0]):
                        input_img = images_ref_input[k:k+1]# type: ignore
                        sample_images = infer.sample(
                            cond_scale=config.checkpoint.cond_scale,
                            texts = texts_ref,
                            start_image_or_video=input_img,
                            start_at_unet_number = 2,
                        ).detach().cpu() # B x C x H x W
                        image_list.append(sample_images[0])

                sample_images=torch.stack(image_list)
                input_img=images_ref_input
                input_img=input_img.permute(2, 3, 0,1)
                sample_images=sample_images.permute(2, 3, 0,1)
                affine = np.eye(4)  # example affine matrix
                nii = nib.Nifti1Image(sample_images.numpy(), affine)
                nib.save(nii, os.path.join(save_folder, "images", f"sample-{path_name}.nii.gz"))

            infer.accelerator.wait_for_everyone()

            additional_data = {
                "time_elapsed": time.time() - start_time,
            }

