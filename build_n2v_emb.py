import torch
import src.utils as utils

from tqdm import tqdm
import os
from os import listdir
from os.path import isfile, join


n2v_dir = "n2v_emb"
n2v_dim = 64

# Creates the folder n2v_dir/n2v_dim
node2vec_dir = os.path.join(n2v_dir, n2v_dim)
os.makedirs(node2vec_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

files = sorted([f for f in listdir('data/sat_rand') if isfile(join('data/sat_rand', f))])

for f in tqdm(files):
    dimacs_path = join('data/sat_rand', f)
        
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
                                file_name=f,
                                num_workers=0,
                                verbose=1)

                