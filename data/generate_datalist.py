import os
import pdb

root = 'church_lsun'

image_file_list = []

for k in os.listdir(root):
    if os.path.isdir(os.path.join(root,k)):
        for jj in os.listdir(os.path.join(root,k)):
            if jj.split('.')[-1]=='jpg':
                image_file_list.append(os.path.join(k,jj)+'\n')

with open(os.path.join(root,'image_list.txt'), 'w') as f:
    for line in image_file_list:
        f.write(line)
print(f'image_list.txt saved in {root}')
