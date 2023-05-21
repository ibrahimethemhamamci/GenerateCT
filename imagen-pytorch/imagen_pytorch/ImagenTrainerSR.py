import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from einops import rearrange
from accelerate import Accelerator
from pathlib import Path
from shutil import rmtree

# Helper functions
def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log


class ImagenTrainerSR(nn.Module):
    def __init__(
        self,
        imagen,
        *,
        num_train_steps,
        batch_size,
        dataset,
        lr=3e-4,
        wd=0.,
        max_grad_norm=0.5,
        save_results_every=100,
        save_model_every=1000,
        results_folder='./results',
        accelerate_kwargs: dict = dict(),
    ):
        super().__init__()

        self.accelerator = Accelerator(**accelerate_kwargs)

        self.imagen = imagen
        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size

        all_parameters = set(imagen.parameters())

        self.optim = torch.optim.Adam(all_parameters, lr=lr, weight_decay=wd)

        self.max_grad_norm = max_grad_norm

        # Split dataset into train and validation sets
        train_len = int(len(dataset) * 0.8)
        val_len = len(dataset) - train_len
        self.ds, self.valid_ds = random_split(dataset, [train_len, val_len])

        # DataLoaders
        self.dl = DataLoader(self.ds, batch_size=batch_size)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=batch_size)

        # Prepare with accelerator
        (
            self.imagen,
            self.optim,
            self.dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.imagen,
            self.optim,
            self.dl,
            self.valid_dl
        )

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents=True, exist_ok=True)

    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return

        pkg = dict(
            model=self.accelerator.get_state_dict(self.imagen),
            optim=self.optim.state_dict(),
        )
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path)

        imagen = self.accelerator.unwrap_model(self.imagen)
        imagen.load_state_dict(pkg['model'])

        self.optim.load_state_dict(pkg['optim'])

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process
    
    def train_step(self):
        device = self.device

        steps = int(self.steps.item())

        self.imagen.train()

        # logs
        logs = {}

        # update Imagen model
        videos, texts = next(self.dl_iter)

        videos = videos.to(device)
        texts = list(texts)
        videos = videos

        with self.accelerator.autocast():
            loss=0
            videos.cpu()
            print(videos.shape)
            for i in range(videos.shape[2]):
                print(videos.shape)
                print(i)
                loss = self.imagen(videos[:,:,i,:,:].cuda(), texts=texts, unet_number = 2)

                self.accelerator.backward(loss)

                accum_log(logs, {'loss': loss.item()})
                self.max_grad_norm = 10
                if exists(self.max_grad_norm):
                    self.accelerator.clip_grad_norm_(self.imagen.parameters(),self.max_grad_norm)
                self.optim.step()
                self.optim.zero_grad()

            self.print(f"{steps}: loss: {logs['loss']}")

            self.steps += 1
            return logs

    def train(self, log_fn=noop):
        device = next(self.imagen.parameters()).device
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')


