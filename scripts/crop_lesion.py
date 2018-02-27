import os
import numpy as np
from sympy import Point, Segment
import cv2
import tqdm
from scipy.misc import imread
from scipy.misc import imsave
import pickle
import csv
import argparse
from glob import glob

def get_lines(lines):
    x1 = float(lines[0][0])
    y1 = float(lines[0][1])
    x2 = float(lines[1][0])
    y2 = float(lines[1][1]) 

    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    return x1, y1, x2, y2

def cropCrops(images, target_dir, logfile):
    WIDTH = 500
    HEIGHT = 500
    WHOLE_H = 4000
    WHOLE_W = 6000
    saved = 0
    total_count = 0

    for file_name in tqdm.tqdm(images):
        csv_path = os.getcwd() + "/../data/Count/" + file_name.split(".")[0] + "_results.csv"
        csfile = open(csv_path, 'r')
        csfreader = csv.reader(csfile)
        line_list = []
        rows = []

        for row in csfreader:
            rows.append(row)

        for l in rows[1:]:
            this_line = ((float(l[2]),float(l[3])), (float(l[4]),float(l[5])))
            line_list.append(this_line)

        image_path = os.path.join(os.pardir, "data/Annotated/" + file_name)
        try:
            image = imread(image_path)
        except OSError:
            continue

        line_count = 0
        total_images = 0

        for lines in line_list:

            crop_w = int(np.round(np.random.uniform(low=WIDTH-25, high=WIDTH+25)))
            crop_h = crop_w

            x1, y1, x2, y2 = get_lines(lines)
            l1 = Segment(Point(x1, y1), Point(x2, y2))
            if l1.is_Point: continue
            
            line_length = float(l1.length)

            angle_radians = np.arctan(float(l1.slope))
            angle = np.degrees(angle_radians)
            M = cv2.getRotationMatrix2D((x1, y1), angle, 1)
            rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

            if y1 > crop_h/2:   corner_y = y1 - crop_h/2
            else:               corner_y = y1
            if x1 > crop_w/3:   corner_x = x1 - crop_w/3
            else:               corner_x = x1            

            #while corner_x < x1 + line_length / 2:
            on = True
            while on:
                up_down_offset = int(np.round(np.random.uniform(low=-30, high=30)))
                left_right_offset = int(np.round(np.random.uniform(low=-30, high=30)))

                angle = np.round(np.random.uniform(low=-20, high=20))
                '''
                Matrix = cv2.getRotationMatrix2D((corner_x + crop_h/2, 
                     corner_y + crop_w/2), angle, 1)

                newly_rotated = cv2.warpAffine(rotated, Matrix, 
                     (image.shape[1],image.shape[0]))
		'''
                newly_rotated = image

                end_crop_y = int(corner_y + up_down_offset + crop_h)
                end_crop_x = int(corner_x + left_right_offset + crop_w)
                crop_start_y = int(corner_y + up_down_offset)
                crop_start_x = int(corner_x + left_right_offset)

                if end_crop_y > WHOLE_H:
                    tqdm.tqdm.write("Out of bounds.")
                    end_crop_y = WHOLE_H
                    crop_start_y = end_crop_y - crop_h
                
                if end_crop_x > WHOLE_W:
                    tqdm.tqdm.write("Out of bounds.")
                    end_crop_x = WHOLE_W
                    crop_start_x = end_crop_x - crop_w


                cropped = newly_rotated[crop_start_y : end_crop_y, \
                                        crop_start_x : end_crop_x]

                if cropped.shape == (crop_w, crop_h, 3):
                    new_img_path = target_dir + str(file_name) + \
                                   str(line_count) + 'cropped' + str(corner_x) + \
                                   str(corner_y) + '.jpg'

                    new_img = cv2.resize(cropped, (224, 224), 
                                            interpolation = cv2.INTER_LINEAR)

                    imsave(new_img_path, new_img)

                    total_images += 1

                else:
                    logfile.write(str(file_name))
                    logfile.write(str(cropped.shape))

                # How much do we want to juice out of one lesion? :/
                # corner_x += np.round(line_length / 3.0)
                on = False

            line_count += 1

    tqdm.tqdm.write(str(saved))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=str, default='test')
    args = parser.parse_args()

    source = os.path.dirname(__file__)
    parent = os.path.join(source, '../')
    log_path = os.path.join(parent, "logs/files_note_written_log_train1.txt")
    logfile = open(log_path, "w")
    trains = pickle.load(open('yes_train.pkl', 'rb'))
    vals = pickle.load(open('yes_val.pkl', 'rb'))
    tests = pickle.load(open('yes_test.pkl', 'rb'))

    train_target_dir = '../new_data/train/lesion/'
    val_target_dir = '../new_data/val/lesion/'
    test_target_dir = '../new_data/test/lesion/'

    # if args.part == 'train':
    cropCrops(trains, train_target_dir, logfile)
    # elif args.part == 'val':
    cropCrops(vals, val_target_dir, logfile)
    # elif args.part == 'test':
    cropCrops(tests, test_target_dir, logfile)

    logfile.close()
