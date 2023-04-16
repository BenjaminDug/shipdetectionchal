import argparse
import os
from data.split_data import split
from data.tuilage import make_tile
from data.withdraw import removesomebg
from utils import make_yaml
from ultralytics import YOLO

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--origdata", help="path to the dataset to split, if empty we consider split done", type=str)
    parser.add_argument("--destsplit", help="path where to save images to split", type=str,required=True)
    parser.add_argument("--destdatayolo", help="path where to save tile for yolo", type=str)
    parser.add_argument("--withdraw",
                        help="after tiling, if you have too much background you can withdraw some - plz use ratio of background you want over ship - 0 means no withdraw",
                        type=int, default=0)
    parser.add_argument("--maketile", help="create tile or not before training", type=int,default=1)
    parser.add_argument("--tilesize", help="size of the images", type=int, default=640)
    parser.add_argument("--yolov8path",
                        help="yolov8 model to load - use default yolov8 from ultralytics or one of yours", type=str)
    parser.add_argument("--imgsz", help="image size of yolov8 argument", type=int, default=640)

    args = parser.parse_args()
    origdata = args.origdata
    destsplit = args.destsplit
    destdatayolo = args.destdatayolo
    tilesize = args.tilesize
    maketile = args.maketile
    withdraw = args.withdraw
    yolov8path = args.yolov8path
    imgsz = args.imgsz

    # split data
    if origdata:
        print("splitting data ...")
        split(origdata, destsplit)
        print("done ! ")
    # prepare data
    # create tile and label for training yolo
    train_yolo = os.path.join(destdatayolo, "train")
    val_yolo = os.path.join(destdatayolo, "val")
    train_split = os.path.join(destsplit, "train")
    val_split = os.path.join(destsplit, "val")
    if maketile:
        print(f"creating tile of size {tilesize} ...")
        make_tile(train_yolo, val_yolo, train_split, val_split, origdata)
        print("done ! ")

    if withdraw:
        print(f"revisiting ratio of background vs ship")
        garbagepath = "/tmp/todel"
        impath = os.path.join(train_yolo, "images")
        labelpath = os.path.join(train_yolo, "labels")
        removesomebg(impath, labelpath, garbagepath, ratio=3)
        print("done")

    # train yolo know
    print("Let's go ... ")
    make_yaml(destdatayolo)

    model = YOLO(yolov8path)  # load a pretrained model

    results = model.train(data=os.path.join(destdatayolo, 'train.yaml'),
                          batch=8,
                          epochs=50,
                          name=f'yolo_{imgsz}_{tilesize}',
                          imgsz=imgsz,
                          show=True,
                          )
