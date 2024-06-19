import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--img_dir",
    type=str,
    default=None,
    help="img directory of dataset",
    required=True,
)
parser.add_argument(
    "--mask_dir",
    type=str,
    default=None,
    help="mask directory of dataset",
    required=True,
)
args = parser.parse_args()

img_dir = args.img_dir
mask_dir = args.mask_dir
# root_dir = "/home/xxx/code/CoDeF/all_sequences"
# name = "beauty_1"

# msk_folder = f"{root_dir}/{name}_masks"
# img_folder = f"{root_dir}/{name}"
# frg_mask_folder = f"{root_dir}/{name}_masks_0"
# bkg_mask_folder = f"{root_dir}/{name}_masks_1"
msk_folder = mask_dir
img_folder = img_dir
frg_mask_folder = mask_dir + "_0"
bkg_mask_folder = mask_dir + "_1"
os.makedirs(frg_mask_folder, exist_ok=True)
os.makedirs(bkg_mask_folder, exist_ok=True)

# print(msk_folder)
files = glob(msk_folder + "/*.png")
num = len(files)

for i in tqdm(range(num)):
    file_n = os.path.basename(files[i])
    mask = cv2.imread(os.path.join(msk_folder, file_n), 0)
    mask[mask > 0] = 1
    cv2.imwrite(os.path.join(frg_mask_folder, file_n), mask * 255)

    bg_mask = mask.copy()
    bg_mask[bg_mask == 0] = 127
    bg_mask[bg_mask == 255] = 0
    bg_mask[bg_mask == 127] = 255
    cv2.imwrite(os.path.join(bkg_mask_folder, file_n), bg_mask)
