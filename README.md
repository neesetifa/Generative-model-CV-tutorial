# Generative model tutorial
### Theory
VAE https://hip-cuckoo-8c7.notion.site/VAE-14225ee585b5804d9ceccca43f055f99 <br>
Flow https://hip-cuckoo-8c7.notion.site/Flow-14225ee585b580b0b4d4defa8ece6cc2 <br>
Diffusion https://hip-cuckoo-8c7.notion.site/Denoising-Diffusion-Model-DDM-14225ee585b58051b2ebff48025ddb90 <br>
I suggest checking VAE first then go to Diffusion model
### Prepare training data
Training Dataset can be found here: http://dl.yf.io/lsun/<br> 
After downloaded it, unzip it and run data/extract_data.py to extract jpg file from lmdb file.<br>
### Step into code
Currently only Diffusion is available. VAE and Flow are on the way.
### Play with pretrained model
There is a pretrained weight in release. This weight is trained from scratch on church dataset.
### Train model
Training ddpm:  python ddpm.py<br>
