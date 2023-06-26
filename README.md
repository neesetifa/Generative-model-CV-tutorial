# Naive_Denosing_Diffusion
### Prepare training data
Training Dataset can be found here: http://dl.yf.io/lsun/<br> 
After downloaded it, unzip it and run data/extract_data.py to extract jpg file from lmdb file.<br>
### Play with pretrained model
There is a pretrained weight in release. This weight is trained from scratch on church dataset.
### Results with DDPM
![image](https://github.com/neesetifa/Naive_Denosing_Diffusion/blob/main/sample_results/church_result_1_ddpm.jpg)
### Train model
Training ddpm:  python ddpm.py<br>
