import glob
import random
import os
from tqdm import tqdm


def removesomebg(impath, labelpath, garbagepath, ratio):
    os.makedirs(garbagepath, exist_ok=True)

    im_file = glob.glob(impath + "/*png")
    lab_file = glob.glob(labelpath + "/*txt")

    nb_ship = len(lab_file)
    nb_back = len(im_file) - nb_ship

    print(f'there are {len(im_file)} images but {nb_ship} with ships')

    filename_with_ship = [lab.split('/')[-1].split('.')[0] for lab in lab_file]
    random.shuffle(im_file)

    over = False
    for file in tqdm(im_file):
        filename = file.split('/')[-1].split('.')[0]
        if filename not in filename_with_ship:
            os.system(f"mv {file} {garbagepath}")
            nb_back -= 1
            if nb_back == nb_ship * ratio:  # condition to know how many bg we keep
                over = True
                break
        if over:
            break


if __name__ == '__main__':
    impath = "pathtoyourproject/shipdetection/data/yolo_format_640_recovermore/train/images"
    labelpath = "pathtoyourproject/shipdetection/data/yolo_format_640_recovermore/train/labels"
    garbagepath = "pathtoyourproject/shipdetection/data/back_640_more"

    removesomebg(impath, labelpath, garbagepath, ratio=3)
