from __future__ import print_function
import numpy as np
import cv2
import glob
from scipy.misc import imread, imsave

alpha = .5
count = 10
# load the image
heatmap_path = 'heatmaps/models/FriMar2314:41:372018'
image_path = 'data/Annotated/'
save_path = 'overlays/'

images = glob.glob(heatmap_path + '/*.jpg')
for hm_path in sorted(images)[:count]:
	img_name = hm_path.split('/')[-1]
	print(image_path + img_name)
	image = imread(image_path + img_name)
	output = image.copy()
	heatmap = imread(hm_path)
	scaled_hm = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
	scaled_hm = np.repeat(scaled_hm[:,:,np.newaxis], 3, axis=2)
	scaled_hm = cv2.bitwise_not(scaled_hm)
	cv2.addWeighted(scaled_hm, alpha, output, 1 - alpha, 0, output)
	imsave(save_path + img_name, output)
