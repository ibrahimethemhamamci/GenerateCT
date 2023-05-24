# GenerateCT: Text-Guided 3D Chest CT Generation

Welcome to the official repository of GenerateCT, a pioneering work in text-conditional 3D medical image generation, with a particular focus on chest CT volumes. GenerateCT provides an open-source codebase, pre-trained models for text-to-CT generation, and a unique, generated dataset, all freely accessible for researchers.

<p align="center">
  <img src="figures/examples.gif" width="100%">
</p>

## Requirements

Before you start, you must install the necessary dependencies. To do so, execute the following commands:

```setup
# Navigate to the 'super_resolution' directory and install the required packages
cd super_resolution
pip install -e .

# Return to the root directory
cd ..

# Navigate to the 'transformer_maskgit' directory and install its required packages
cd transformer_maskgit
pip install -e .

# Return to the root directory
cd ..
```
After following these steps, your environment should be properly set up with all required packages.

## Training

Train the CT-ViT model by executing the following command in your terminal:

```train
accelerate launch --use_fsdp train_ctvit.py
```
To train the MaskGIT Transformer model, use the command provided below:

```train
accelerate launch train_transformer.py
```

Lastly, train the Super Resolution Diffusion model using the multi-line command outlined here:

```train
accelerate launch \
    --multi_gpu \
    --mixed_precision=fp16 \
    --num_machines=1 \
    train_superres.py --config superres.yaml --stage 2 --bs 8
```
Please make sure to adjust the parameters based on your setup's requirements. Should you encounter any issues during the training process, feel free to raise an issue on our GitHub page.

## Inference

For inference of the CT-ViT model in the paper, run this command:

```eval
python inference_ctvit.py
```

For inference of the MaskGIT Transformer model in the paper, run this command:

```eval
python inference_transformer.py
```

For inference of the Super Resolution Diffusion model, run this command:

```eval
accelerate launch \
    --multi_gpu \
    --mixed_precision=fp16 \
    --num_machines=1 \
    inference_superres.py --config superres_inference.yaml --stage 2 --bs 2
```

## Pretrained Models

You can download pretrained models here:

- Pretrained models -- [CT-ViT](https://huggingface.co/generatect/GenerateCT/resolve/main/pretrained_models/ctvit_pretrained.pt), [Transformer](https://huggingface.co/generatect/GenerateCT/resolve/main/pretrained_models/transformer_pretrained.pt), and [Super Resolution Diffusion](https://huggingface.co/generatect/GenerateCT/resolve/main/pretrained_models/superres_pretrained.pt) -- trained on our paired radiological report&chest CT volume dataset described in the paper. 

## Example Data

You can download example data here:

- [Example data](https://huggingface.co/generatect/GenerateCT/resolve/main/example_data.zip) for the CT-ViT, transformer, and Super Resolution Diffusion networks' trainings. 

## Generated Dataset

You can download generated dataset with 2286 generated CT volumes and their corresponding text prompts here:

- [Generated dataset](https://huggingface.co/generatect/GenerateCT/tree/main/generated_data), used in the supplementary section for a clinical application to demonstrate the potential of GenerateCT, consisting synthetic chest CT volumes, medical language text prompts used in generation, and abnormality labels. 

## Evaluation

For FID and FVD, we used dataset evaluation script from the [StyleGAN-V](https://github.com/universome/stylegan-v). For the CLIP score, we used [torchmetrics implementation](https://torchmetrics.readthedocs.io/en/stable/multimodal/clip_score.html).


## Results

Our models achieve the following performances :

|                    |   Resolution    |    Dimension     |    Text-Guided   |        FID (↓)               |  FVD (↓)  |  CLIP (↑)   |
|--------------------|-----------------|------------------|------------------|------------------------------|-----------|-------------|
|      CT-ViT        |       128       |       3D         |        No        |         73.4                 |   1817.4  |    N/A      |
|   Transformer      |       128       |       3D         |        Yes       |        104.3                 |   1886.8  |    25.2     |
|     Diffusion      |       512       |       2D         |        Yes       |         14.9                 |   409.8   |    27.6     |
|   **GenerateCT**   |       512       |       3D         |        Yes       |         55.8                 |   1092.3  |    27.1     |



## License
Our work, including the codes, trained models, and generated data, is released under a [Creative Commons Attribution (CC-BY) license](https://creativecommons.org/licenses/by/4.0/). This means that anyone is free to share (copy and redistribute the material in any medium or format) and adapt (remix, transform, and build upon the material) for any purpose, even commercially, as long as appropriate credit is given, a link to the license is provided, and any changes that were made are indicated. This aligns with our goal of facilitating progress in the field by providing a resource for researchers to build upon. 


## Acknowledgements
We would like to express our gratitude to the following repositories for their invaluable contributions to our work: [Phenaki Pytorch by Lucidrains](https://github.com/lucidrains/phenaki-pytorch), [Phenaki by LAION-AI](https://github.com/LAION-AI/phenaki), [Imagen Pytorch by Lucidrains](https://github.com/lucidrains/imagen-pytorch), [StyleGAN-V by universome](https://github.com/universome/stylegan-v), and [CT Net Models by Rachellea](https://github.com/rachellea/ct-net-models). We extend our sincere appreciation to these researchers for their exceptional open-source efforts. If you utilize our models and code, we kindly request that you also consider citing these works to acknowledge their contributions.

