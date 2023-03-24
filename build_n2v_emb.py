import torch
import src.utils as utils

from tqdm import tqdm
import os
from os import listdir
from os.path import isfile, join


data_path = 'data/rand/0050'
n2v_dir = "n2v_emb"
n2v_dim = 64

# Creates the folder n2v_dir/n2v_dim
node2vec_dir = os.path.join(n2v_dir, str(n2v_dim))
os.makedirs(node2vec_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

paths = []
for root, dirs, files in os.walk(data_path):
    for filename in files:
        paths.append(os.path.join(root, filename))
paths = sorted(paths)

for dimacs_path in tqdm(paths):
    tail = os.path.split(dimacs_path)[1]  # returns filename and extension
    node2vec_filename = os.path.splitext(tail)[0]  # returns filename

    _ = utils.node2vec(dimacs_path,
                       device,
                       embedding_dim=n2v_dim,
                       walk_length=10,
                       context_size=5,
                       walks_per_node=5,
                       p=1,
                       q=1,
                       batch_size=32,
                       lr=0.01,
                       num_epochs=150,
                       save_path=node2vec_dir,
                       file_name=node2vec_filename,
                       num_workers=0,
                       verbose=1)
    