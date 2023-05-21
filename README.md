# GenerateCT: Text-Guided 3D Chest CT Generation

Welcome to the official repository of GenerateCT, a pioneering work in text-conditional 3D medical image generation.

This repository contains the official implementation of GenerateCT, a cutting-edge method that leverages text guidance to generate high-quality 3D medical images, with a particular focus on chest CT volumes.


Feel free to explore the code, experiment with the models, and contribute to the advancement of this exciting field!


>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our models achieve the following performances :

|                    |   Resolution    |    Dimension     |    Text-Guided   |        FID (↓)               |  FVD (↓)  |  CLIP (↑)   |
|--------------------|-----------------|------------------|------------------|------------------------------|-----------|-------------|
|      CT-ViT        |       128       |       3D         |        No        |         73.4                 |   1817.4  |    N/A      |
|   Transformer      |       128       |       3D         |        Yes       |        104.3                 |   1886.8  |    25.2     |
|     Diffusion      |       512       |       2D         |        Yes       |         14.9                 |   409.8   |    27.6     |
|   **GenerateCT**   |       512       |       3D         |        Yes       |         55.8                 |   1092.3  |    27.1     |


>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing
Our work, including the codes, trained models, and generated data, is released under a [Creative Commons Attribution (CC-BY) license](https://creativecommons.org/licenses/by/4.0/). This means that anyone is free to share (copy and redistribute the material in any medium or format) and adapt (remix, transform, and build upon the material) for any purpose, even commercially, as long as appropriate credit is given, a link to the license is provided, and any changes that were made are indicated. This aligns with our goal of facilitating progress in the field by providing a resource for researchers to build upon. 


