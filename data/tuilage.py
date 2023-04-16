from data.datautils import tuile_convertYOLO, prepare_dataframe
from tqdm import tqdm
import pandas as pd
import os


def make_tile(train_yolo:str, val_yolo:str, train_split:str, val_split:str, orig_path:str,tilesize:int):
    os.system(f'mkdir -p {train_yolo}/images')
    os.system(f'mkdir -p {train_yolo}/labels')
    os.system(f'mkdir -p {val_yolo}/images')
    os.system(f'mkdir -p {val_yolo}/labels')

    df = prepare_dataframe(orig_path)

    train_list = os.listdir(train_split)
    val_list = os.listdir(val_split)

    print("make train")
    for filetrain in tqdm(train_list):
        tuile_convertYOLO(filename=filetrain,
                          path=train_split,
                          dest=train_yolo,
                          size=tilesize,
                          df=df,
                          recover=0.5,
                          RGB=True)

    for fileval in tqdm(val_list):
        tuile_convertYOLO(filename=fileval,
                          path=val_split,
                          dest=val_yolo,
                          size=tilesize,
                          df=df,
                          recover=0.5,
                          RGB=True)


if __name__ == "__main__":
    train_yolo = "pathtoyourproject/shipdetection/data/yolo_format_640_recovermore/train"
    val_yolo = "pathtoyourproject/shipdetection/data/yolo_format_640_recovermore/val"

    train_split = "pathtoyourproject/shipdetection/data/split/train"
    val_split = "pathtoyourproject/shipdetection/data/split/val"
    orig_path = "pathtoyourproject/shipdetection/data/original/train/"

    make_tile(train_yolo, val_yolo, train_split, val_split, orig_path,tilesize=640)
