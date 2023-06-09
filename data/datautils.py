from tqdm import tqdm
import random
import torch
import json
import os
import cv2
import glob
import pandas as pd
from PIL import Image
from typing import Tuple,List

def prepare_dataframe(originpath:str):
    imtrain = glob.glob(os.path.join(originpath, "*png"))
    print(f"there are {len(imtrain)} images ")

    with open(os.path.join(originpath, "metadata.jsonl"), 'r') as jsonfile:
        json_list = list(jsonfile)

    filename = []
    bbox = []
    categ = []
    for file in json_list:
        loadfile = json.loads(file)
        filename.append(loadfile['file_name'])
        bbox.append(loadfile['objects']['bbox'])
        categ.append(loadfile['objects']['categories'])

    df_train = pd.DataFrame([filename, bbox, categ]).T
    df_train.columns = ['filename', 'bbox', 'categ']
    df_train = df_train.explode(column=['bbox', 'categ']).reset_index(drop=True)
    print(df_train.head())

    hauteur = []
    largeur = []
    nom = []
    allfiles = df_train.filename.unique()
    for file in tqdm(allfiles):
        img = cv2.imread(os.path.join(originpath, file))
        nom.append(file)
        hauteur.append(img.shape[0])
        largeur.append(img.shape[1])

    tmp = pd.DataFrame([nom, hauteur, largeur]).T
    tmp.columns = ['filename', 'hauteur', 'largeur']
    df_train_m = df_train.merge(tmp, on='filename', how='left')
    print(tmp.describe())
    print("***************************************************\n\n\n")
    print(df_train_m.head(30))

    return df_train_m


def dota2yolo(x1, y1, x2, y2, W, H):
    x = (x2 + x1) // 2
    y = (y2 + y1) // 2
    w = x2 - x1
    h = y2 - y1
    return x / W, y / H, w / W, h / H


def yolo2dota(x, y, w, h, W, H):
    x = x * W
    w = w * W
    y = y * H
    h = h * H

    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    return x1, y1, x2, y2


def keeplabel(rangepropose:Tuple, label:List, tresh=0.5):
    """
    We only keep labels where x,y are in the picture. If 50% of aera of bounding box is out of picture we do not label

    """
    x1, y1, x2, y2 = label
    w = x2 - x1
    h = y2 - y1
    xmin, xmax, ymin, ymax = rangepropose

    if x1 < xmin or x1 > xmax or y1 < ymin or y1 > ymax:
        return 0
    if (x1 + w * tresh) > xmax or (y1 + h * tresh) > ymax:
        return 0

    return 1


def newlabel(rangepropose:Tuple, label:List):
    """
    we need label adapted for the tile

    """
    x1, y1, x2, y2 = label
    xmin, xmax, ymin, ymax = rangepropose
    return x1 - xmin, y1 - ymin, x2 - xmin, y2 - ymin


def get_bbox(df:pd.DataFrame, filename:str):
    return df[df['filename'] == filename].bbox.to_list()


def tuile_convertYOLO_old(filename:str, path:str, dest:str, size:int, df:pd.DataFrame, recover:float=0.5,RGB=True):
    """
     making tile from big picture
    """
    img = cv2.imread(os.path.join(path, filename))
    if RGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    destim = os.path.join(dest, 'images')
    destlab = os.path.join(dest, 'labels')

    Ylen, Xlen, channel = img.shape
    list_tile = []
    bbox = get_bbox(df, filename)
    over=False
    if size < Xlen and size < Ylen:
        for nb_Y in range(0, Ylen, int(size * recover)):
            for nb_X in range(0, Xlen, int(size * recover)):
                if nb_X + size < Xlen and nb_Y + size < Ylen:
                    ima_tile = img[nb_Y:nb_Y + size, nb_X:nb_X + size, :]
                    name_imatile = filename.split('.')[0] + f"_{nb_X}_{nb_Y}"
                    list_tile.append(ima_tile)
                    for box in bbox:
                        if keeplabel((nb_X, nb_X + size, nb_Y, nb_Y + size), box, tresh=0.25):
                            adaptedbox = newlabel((nb_X, nb_X + size, nb_Y, nb_Y + size), box)
                            x, y, w, h = dota2yolo(*adaptedbox, W=ima_tile.shape[1], H=ima_tile.shape[0])
                            with open(os.path.join(destlab, name_imatile+".txt"), 'a') as f:
                                f.write(f"{0} {x} {y} {w} {h}\n")

                    cv2.imwrite(os.path.join(destim, name_imatile+".png"), ima_tile)
                elif nb_X + size > Xlen and nb_Y + size > Ylen:
                    ima_tile = img[Ylen - size:Ylen, Xlen - size:Xlen, :]
                    name_imatile = filename.split('.')[0] + f"_{nb_X}_{nb_Y}"
                    list_tile.append(ima_tile)
                    for box in bbox:
                        if keeplabel((nb_X, nb_X + size, nb_Y, nb_Y + size), box, tresh=0.25):
                            adaptedbox = newlabel((nb_X, nb_X + size, nb_Y, nb_Y + size), box)
                            x, y, w, h = dota2yolo(*adaptedbox, W=ima_tile.shape[1], H=ima_tile.shape[0])
                            with open(os.path.join(destlab, name_imatile + ".txt"), 'a') as f:
                                f.write(f"{0} {x} {y} {w} {h}\n")

                    cv2.imwrite(os.path.join(destim, name_imatile + ".png"), ima_tile)
                    over = True
                    break

                elif nb_X + size > Xlen:
                    ima_tile = img[nb_Y:nb_Y + size, Xlen - size:Xlen, :]
                    name_imatile = filename.split('.')[0] + f"_{nb_X}_{nb_Y}"
                    list_tile.append(ima_tile)
                    for box in bbox:
                        if keeplabel((nb_X, nb_X + size, nb_Y, nb_Y + size), box, tresh=0.25):
                            adaptedbox = newlabel((nb_X, nb_X + size, nb_Y, nb_Y + size), box)
                            x, y, w, h = dota2yolo(*adaptedbox, W=ima_tile.shape[1], H=ima_tile.shape[0])
                            with open(os.path.join(destlab, name_imatile + ".txt"), 'a') as f:
                                f.write(f"{0} {x} {y} {w} {h}\n")

                    cv2.imwrite(os.path.join(destim, name_imatile + ".png"), ima_tile)
                    break
                elif nb_Y + size > Ylen:
                    ima_tile = img[Ylen - size:Ylen, nb_X:nb_X + size, :]
                    name_imatile = filename.split('.')[0] + f"_{nb_X}_{nb_Y}"
                    list_tile.append(ima_tile)
                    for box in bbox:
                        if keeplabel((nb_X, nb_X + size, nb_Y, nb_Y + size), box, tresh=0.25):
                            adaptedbox = newlabel((nb_X, nb_X + size, nb_Y, nb_Y + size), box)
                            x, y, w, h = dota2yolo(*adaptedbox, W=ima_tile.shape[1], H=ima_tile.shape[0])
                            with open(os.path.join(destlab, name_imatile + ".txt"), 'a') as f:
                                f.write(f"{0} {x} {y} {w} {h}\n")

                    cv2.imwrite(os.path.join(destim, name_imatile + ".png"), ima_tile)
            if over:
                break

    elif size > Xlen and size > Ylen:
        print(f"size of the img {filename} is to small: {img.shape}")
        print("neither one take it all")
        cv2.imwrite(os.path.join(destim, filename), img)
        print("bbox: ",bbox)
        for box in bbox:
            x, y, w, h = dota2yolo(*box, W=img.shape[1], H=img.shape[0])
            with open(os.path.join(destlab, filename.split('.')[0] + ".txt"), 'a') as f:
                f.write(f"{0} {x} {y} {w} {h}\n")


    elif Xlen>size:
        print(f"Xlen>size {Xlen>size} ")
        for nb_X in range(0, Xlen, int(size * recover)):
            if nb_X + size>Xlen:
                ima_tile = img[:, nb_X-size:Xlen, :]
                name_imatile = filename.split('.')[0] + f"_{nb_X}_{0}"
                for box in bbox:
                    if keeplabel((nb_X, nb_X + size, 0, Ylen), box, tresh=0.25):
                        adaptedbox = newlabel((nb_X, nb_X + size, 0, Ylen), box)
                        x, y, w, h = dota2yolo(*adaptedbox, W=ima_tile.shape[1], H=ima_tile.shape[0])
                        with open(os.path.join(destlab, name_imatile + ".txt"), 'a') as f:
                            f.write(f"{0} {x} {y} {w} {h}\n")
                cv2.imwrite(os.path.join(destim, name_imatile + ".png"), ima_tile)
                break
            ima_tile = img[:, nb_X:nb_X + size, :]
            name_imatile = filename.split('.')[0] + f"_{nb_X}_{0}"
            for box in bbox:
                if keeplabel((nb_X, nb_X + size, 0, Ylen), box, tresh=0.25):
                    adaptedbox = newlabel((nb_X, nb_X + size, 0, Ylen), box)
                    x, y, w, h = dota2yolo(*adaptedbox, W=ima_tile.shape[1], H=ima_tile.shape[0])
                    with open(os.path.join(destlab, name_imatile+".txt"), 'a') as f:
                        f.write(f"{0} {x} {y} {w} {h}\n")
            cv2.imwrite(os.path.join(destim, name_imatile+".png"), ima_tile)

    elif Ylen > size:

        print(f"Ylen>size {Ylen > size} ")

        for nb_Y in range(0, Ylen, int(size * recover)):

            if nb_Y + size > Ylen:

                ima_tile = img[Ylen - size:Ylen, :, :]

                name_imatile = filename.split('.')[0] + f"_{0}_{nb_Y}"

                for box in bbox:

                    if keeplabel((0, Xlen, nb_Y, nb_Y + size), box, tresh=0.25):
                        adaptedbox = newlabel((0, Xlen, nb_Y, nb_Y + size), box)

                        x, y, w, h = dota2yolo(*adaptedbox, W=ima_tile.shape[1], H=ima_tile.shape[0])

                        with open(os.path.join(destlab, name_imatile + ".txt"), 'a') as f:
                            f.write(f"{0} {x} {y} {w} {h}\n")

                cv2.imwrite(os.path.join(destim, name_imatile + ".png"), ima_tile)

                break

            ima_tile = img[nb_Y:nb_Y + size, :, :]

            name_imatile = filename.split('.')[0] + f"_{0}_{nb_Y}"

            for box in bbox:

                if keeplabel((0, Xlen, nb_Y, nb_Y + size), box, tresh=0.25):
                    adaptedbox = newlabel((0, Xlen, nb_Y, nb_Y + size), box)

                    x, y, w, h = dota2yolo(*adaptedbox, W=ima_tile.shape[1], H=ima_tile.shape[0])

                    with open(os.path.join(destlab, name_imatile + ".txt"), 'a') as f:
                        f.write(f"{0} {x} {y} {w} {h}\n")

            cv2.imwrite(os.path.join(destim, name_imatile + ".png"), ima_tile)


def tuile_convertYOLO(filename:str, path:str, dest:str, size:int, df:pd.DataFrame, recover:float=0.5,RGB=True):
    """
    we need label adapted for the tile
    """
    img = cv2.imread(os.path.join(path, filename))
    if RGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    destim = os.path.join(dest, 'images')
    destlab = os.path.join(dest, 'labels')

    Ylen, Xlen, channel = img.shape
    list_tile = []
    bbox = get_bbox(df, filename)
    for nb_X in range(0, Xlen, int(size * recover)):
        for nb_Y in range(0, Ylen, int(size * recover)):
            if nb_X + size < Xlen and nb_Y + size < Ylen:
                ima_tile = img[nb_Y:nb_Y + size, nb_X:nb_X + size, :]
                name_imatile = filename.split('.')[0] + f"_{nb_X}_{nb_Y}"
                list_tile.append(ima_tile)
                for box in bbox:
                    if keeplabel((nb_X, nb_X + size, nb_Y, nb_Y + size), box, tresh=0.25):
                        adaptedbox = newlabel((nb_X, nb_X + size, nb_Y, nb_Y + size), box)
                        x, y, w, h = dota2yolo(*adaptedbox, W=ima_tile.shape[1], H=ima_tile.shape[0])
                        with open(os.path.join(destlab, name_imatile+".txt"), 'a') as f:
                            f.write(f"{0} {x} {y} {w} {h}\n")

                cv2.imwrite(os.path.join(destim, name_imatile+".png"), ima_tile)

    if not list_tile:
        print(f"size of the img {filename} is to small: {img.shape}")
        if Xlen>size:
            print(f"Xlen>size {Xlen>size} ")
            for nb_X in range(0, Xlen, int(size * recover)):
                ima_tile = img[:, nb_X:nb_X + size, :]
                name_imatile = filename.split('.')[0] + f"_{nb_X}_{0}"
                for box in bbox:
                    if keeplabel((nb_X, nb_X + size, 0, Ylen), box, tresh=0.25):
                        adaptedbox = newlabel((nb_X, nb_X + size, 0, Ylen), box)
                        x, y, w, h = dota2yolo(*adaptedbox, W=ima_tile.shape[1], H=ima_tile.shape[0])
                        with open(os.path.join(destlab, name_imatile+".txt"), 'a') as f:
                            f.write(f"{0} {x} {y} {w} {h}\n")
                cv2.imwrite(os.path.join(destim, name_imatile+".png"), ima_tile)

        elif Ylen>size:
            print(f"Ylen>size {Ylen>size} ")
            for nb_Y in range(0, Ylen, int(size * recover)):
                ima_tile = img[nb_Y:nb_Y + size,:, :]
                name_imatile = filename.split('.')[0] + f"_{0}_{nb_Y}"
                for box in bbox:
                    if keeplabel((0, Xlen, nb_Y, nb_Y + size), box, tresh=0.25):
                        adaptedbox = newlabel((0, Xlen, nb_Y, nb_Y + size), box)
                        x, y, w, h = dota2yolo(*adaptedbox, W=ima_tile.shape[1], H=ima_tile.shape[0])
                        with open(os.path.join(destlab, name_imatile+".txt"), 'a') as f:
                            f.write(f"{0} {x} {y} {w} {h}\n")
                cv2.imwrite(os.path.join(destim, name_imatile+".png"), ima_tile)

        else:
            print("neither one take it all")
            cv2.imwrite(os.path.join(destim, filename), img)
            print("bbox: ",bbox)
            for box in bbox:
                x, y, w, h = dota2yolo(*box, W=img.shape[1], H=img.shape[0])
                with open(os.path.join(destlab, filename.split('.')[0] + ".txt"), 'a') as f:
                    f.write(f"{0} {x} {y} {w} {h}\n")


class Shipclassifdataset(torch.utils.data.Dataset):
    #for faster rcnn
    def __init__(self, root, transforms,mode):
        self.root = root
        self.mode = mode
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.labs = glob.glob(os.path.join(root, "labels",'*txt'))
        self.imgs = glob.glob(os.path.join(root, "images",'*png'))
        self.files_ship = [img_pth.split("/")[-1].split(".")[0] for img_pth in self.labs]
        random.shuffle(self.imgs)

    def __getitem__(self, idx):
        # load images and masks

        img_path = self.imgs[idx]
        filename = img_path.split("/")[-1].split(".")[0]

        # img = Image.open(os.path.join(self.root,"images",filename+".png")).convert("RGB")

        img = cv2.imread(os.path.join(self.root,"images",filename+".png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if filename  in self.files_ship:
            labels = torch.ones((1,), dtype=torch.int64)
        else:
            labels = torch.zeros((1,), dtype=torch.int64)
        # convert everything into a torch.Tensor

        img = self.transforms[self.mode](image=img)['image']

        return img, labels

    def __len__(self):
        return len(self.imgs)
