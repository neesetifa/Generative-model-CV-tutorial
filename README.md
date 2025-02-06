# Generative Models in Computer Vision, A Practical Tutorial
#### Overview
This repository provides hands-on implementations of generative models in computer vision, covering Variational AutoEncoders (VAEs), Flows, and Denoising Diffusion Models (DDMs). The goal is to offer a clear, structured tutorial with theoretical insights and practical coding examples for those who interested in deep generative models.

#### Theory
Following are the basic theory and original math derivation of all these generative models. I suggest checking VAE first then go to Diffusion model<br>
- [VAE, updated Nov,2024](https://hip-cuckoo-8c7.notion.site/VAE-14225ee585b5804d9ceccca43f055f99) <br>
- [Flow, updated Oct,2024](https://hip-cuckoo-8c7.notion.site/Flow-14225ee585b580b0b4d4defa8ece6cc2) <br>
- [Diffusion, updated Nov,2024](https://hip-cuckoo-8c7.notion.site/Denoising-Diffusion-Model-DDM-14225ee585b58051b2ebff48025ddb90) <br>

#### Prepare training data
Training Dataset can be found [here](http://dl.yf.io/lsun/)<br> 
After downloaded it, unzip it and run data/extract_data.py to extract jpg file from lmdb file.<br>

#### Training a Model
Train a Diffusion Model (DDPM)
```rb
python ddpm.py
```
VAE and Flow are on the way.

#### Generating Images

Once trained, you can generate new images with:
```rb
python generate.py --model diffusion --num_samples 10
```

Using Pre-trained Models
There is a pretrained weight in release. This weight is trained from scratch on church dataset.
