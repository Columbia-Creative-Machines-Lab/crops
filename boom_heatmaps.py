from __future__ import print_function, division
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
from shutil import copyfile
from skimage.util import view_as_windows
from PIL import Image
from scipy.misc import imsave
import glob
import argparse
import cv2
from tqdm import tqdm
import glob

def image_loader(path):
    to_tensor = transforms.ToTensor()
    normalizer = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')
            w, h = img.size
            img = normalizer(to_tensor(img))
            return img

def make_heatmap(model, name, data_dir, step_size, targets):
    counter = 0
    model.train(False)
    softmax = nn.Softmax()

    for file_name in tqdm(sorted(targets)):
        image = image_loader(os.path.join(data_dir, file_name))
        tqdm.write(file_name)
        img = image.numpy()
        img = np.einsum('ijk->jki', img)
        ratio = 224 / 500
        img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        img = np.einsum('jki->ijk', img)
        blocks = view_as_windows(img, window_shape=(3,224,224), step=step_size)
        blocks = blocks.reshape((blocks.shape[1],blocks.shape[2], 3, 224, 224))

        result = np.zeros((blocks.shape[0], blocks.shape[1]))
        for y in range(blocks.shape[0]):
            pictures = blocks[y,:,:,:,:]

            row_of_images = Variable(torch.FloatTensor(pictures).cuda())
            output = model(row_of_images)
            sm_output = softmax(output)
            predictions = sm_output[:,1].cpu().data.numpy()
            predictions = predictions.reshape(blocks.shape[1])
            result[y,:] = predictions

        imsave('./heatmaps/' + name + file_name, result)
        counter += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="data/Annotated")
    parser.add_argument("--step", type=int, default=60)
    parser.add_argument("--pickle", type=str)
    args = parser.parse_args()
    targets = pickle.load(open(args.pickle, 'rb'))

    model = models.resnet34(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model_list = glob.glob('models/*')
    model_dict = {x: model_list[x] for x in range(len(model_list))}
    print(model_dict)
    idx = int(input("Choose a model"))
    state_dict = torch.load(model_dict[idx] + '/boom_transfer.pymodel')
    model.load_state_dict(state_dict)
    criterion = nn.CrossEntropyLoss()
    model = model.cuda()
    make_heatmap(model, model_dict[idx] + '/', args.dir, args.step, targets)
