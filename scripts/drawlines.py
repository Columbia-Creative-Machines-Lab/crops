import os
import numpy as np
import scipy
import cv2
import scipy.misc
from scipy.misc import imread
from scipy.misc import imsave
import pickle
import csv

def draw_lines(pic_descriptions, target_directory, logfile):
	cwd = os.getcwd()

	mod_matrix = dict()
	mod_matrix[0] = (0, 0)
	mod_matrix[1] = (-3000, 0)
	mod_matrix[2] = (-3000, -2000)
	mod_matrix[3] = (0, -2000)


	for pic in pic_descriptions:
		print pic[0] + "---" + str(pic[-2])

		csfile = open(cwd + pic[1] + pic[2], 'rb')
		csfreader = csv.reader(csfile)
		rows = []
		line_list = []
		file_name = pic[0]
		for row in csfreader:
			rows.append(row)

		for l in rows[1:]:
		 	this_line = [l[2], l[3], l[4], l[5]]
		 	this_line = [int(e) for e in this_line]
		 	line_list.append(this_line)

		img = imread(cwd + '/boom_all_images/train/withlesion/' + pic[0] + '_{}'.format(pic[-2]) + '.jpg')

		line_count = 0
		x, y = mod_matrix[pic[-2]]

		for l in line_list:
			print (l[0], l[1]), (l[2], l[3])

			img = cv2.line(img, (l[0] + x, l[1] + y), (l[2] + x, l[3] + y), (0, 0, 0), 5)
			imsave(target_directory + str(file_name) + '_{}'.format(pic[-2]) + '.jpg', img)
			logfile.write(str(file_name))
			logfile.write(str(img.shape))
			line_count += 1

if __name__ == "__main__":
	with open("train_lesion_boom2.txt", "rb") as fp:
		train_lesion_pics = pickle.load(fp)

	# with open("valid_lesion_boom2.txt", "rb") as fp:
	# 	valid_lesion_pics = pickle.load(fp)

	logfile = open("draw_log_train1.txt", "w")
	train_target_dir = os.getcwd() + '/drawn_lines/'
	draw_lines(train_lesion_pics[:1800], train_target_dir, logfile)
	logfile.close()