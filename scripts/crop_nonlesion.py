import os
import numpy as np
import cv2
from scipy.misc import imread
from scipy.misc import imsave
import csv
import argparse
import glob
import pickle
import tqdm

def cropNL(image_dir, image_names, target_dir, total):
    WIDTH = 500
    HEIGHT = 500
    WHOLE_H = 4000
    WHOLE_W = 6000

    total_images = 0
    num_per_sample = int(total / len(image_names))

    for pic_name in tqdm.tqdm(image_names):
        try:
            original = imread(image_dir + "/" + pic_name)
        except OSError:
            print(pic_name)
            continue
        for idx in tqdm.trange(num_per_sample):
            crop_x = int(np.round(np.random.uniform(low=WIDTH-25, high=WIDTH+25)))
            crop_y = crop_x

            start_x = int(np.round(np.random.uniform(low=0, high=WHOLE_W-crop_x-1)))
            start_y = int(np.round(np.random.uniform(low=0, high=WHOLE_H-crop_y-1)))

            end_x = start_x + crop_x
            end_y = start_y + crop_y

            # angle = np.round(np.random.uniform(low=0, high=359))

            # Matrix = cv2.getRotationMatrix2D((start_x + crop_x/2, 
            #     start_y + crop_y/2), angle, 1)

            # newly_rotated = cv2.warpAffine(original, Matrix, 
            #     (original.shape[1],original.shape[0]))

            newly_rotated = original

            unscaled_pic = newly_rotated[start_y : end_y, start_x : end_x]

            new_pic = cv2.resize(unscaled_pic, (224, 224), 
                                     interpolation = cv2.INTER_LINEAR)

            imsave(target_dir + pic_name.split(".")[0] + "-" + str(crop_x) + ".jpg", new_pic)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="../data/Annotated/")
    args = parser.parse_args()

    trains = pickle.load(open('no_train.pkl', 'rb'))
    vals = pickle.load(open('no_val.pkl', 'rb'))
    tests = pickle.load(open('no_test.pkl', 'rb'))

    train_target_dir = os.getcwd() + '/../hnm_data/'
    #val_target_dir = os.getcwd() + '/../new_data/val/nonlesion/'
    #test_target_dir = os.getcwd() + '/../new_data/test/nonlesion/'

    cropNL(args.dir, trains, train_target_dir, 80000)
    #cropNL(args.dir, vals, val_target_dir, 15000)
    #cropNL(args.dir, tests, test_target_dir, 15000)
