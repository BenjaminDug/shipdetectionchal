import glob
from tqdm import tqdm
import shutil
import os
import random

def split(orig_path:str,dest:str):


    orig_files = glob.glob(os.path.join(orig_path, '*.png'))
    random.shuffle(orig_files)
    nb_fichier = len(orig_files)
    print(f"there are {nb_fichier} images")

    train = int(0.9 * nb_fichier)
    val = int(0.1 * nb_fichier)

    traindir = os.path.join(dest, "train")
    os.system(f'mkdir -p {traindir}')
    for k in tqdm(range(train)):

        shutil.copy(orig_files[k], os.path.join(traindir, orig_files[k].split("/")[-1]))

    valdir = os.path.join(dest, "val")
    os.system(f'mkdir -p {valdir}')
    for k in tqdm(range(train, train + val)):

        shutil.copy(orig_files[k], os.path.join(valdir, orig_files[k].split("/")[-1]))


if __name__ == "__main__":

    orig_path = "pathtoyourproject/shipdetection/data/original/train/"
    dest = "pathtoyourproject/shipdetection/data/split/"

    split(orig_path, dest)
