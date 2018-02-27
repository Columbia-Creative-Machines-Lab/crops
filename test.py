import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from ImageFolder import ImageFolder
import matplotlib.pyplot as plt
from time import time
import copy
import os
from shutil import copyfile
import argparse
from tqdm import tqdm, trange
import IPython
import glob

data_transforms = {
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def test(model, model_name, loss_function=nn.CrossEntropyLoss()):
    running_loss = 0
    running_accuracy = 0
    running_fp = 0
    running_fn = 0
    all_fp = []
    for idx, data in enumerate(tqdm(dataset_loaders['test'], desc="Test")):
        inputs, labels, path = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        output = model(inputs)
        _, predictions = torch.max(output.data, 1)
        loss = loss_function(output, labels)

        running_loss += loss.data[0]
        this_batch_num_acc = torch.sum(predictions == labels.data)
        wrongs = (predictions - labels.data)
        wrongs_idx = wrongs.nonzero()
        wrongs = wrongs.cpu().numpy()
        tqdm.write("{}".format(wrongs))
        wrongs_idx = wrongs_idx.cpu().numpy()
        batchlen = len(labels)
        fp, fn = [],[]
        if len(wrongs_idx) != 0:
            fp = np.where(wrongs==-1)[0]
            fn = np.where(wrongs==1)[0]
        tqdm.write("Batch num acc: {}".format(this_batch_num_acc))
        tqdm.write("{}".format(len(fp)))
        tqdm.write("{}".format(len(fn)))
        fp, fn = list(fp), list(fn)
        paths_fp = [path[m] for m in fp]
        paths_fn = [path[n] for n in fn]
        with open(model_name + '_test_log.txt', 'a+') as f:
            f.write("FP\n")
            f.write(str(paths_fp))
            f.write("\nFN\n")
            f.write(str(paths_fn))
            f.write("\n")
        all_fp = all_fp + paths_fp 
        running_accuracy += this_batch_num_acc
        running_fp += len(fp)
        running_fn += len(fn)

    pickle.dump(all_fp, open('false_positives.pkl', 'wb+'))
    epoch_loss = running_loss / dataset_sizes['test']
    epoch_accuracy = running_accuracy / dataset_sizes['test']

    tqdm.write('Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_accuracy))
    tqdm.write('Fp: {}, Fn: {}'.format(running_fp, running_fn))

if __name__ == "__main__":
    model_list = glob.glob('models/*')
    model_dict = {x: model_list[x] for x in range(len(model_list))}
    print(model_dict)
    idx = int(input("Choose a model"))
    state_dict = torch.load(model_dict[idx] + '/boom_transfer.pymodel')

    model = models.resnet34()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(state_dict)
    model.cuda()
    datasets = {x: ImageFolder(os.path.join("./new_data/", x), data_transforms[x])
         for x in ['test']}
    dataset_loaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=80,
                    shuffle=True, num_workers=4) for x in ['test']}

    dataset_sizes = {x: len(datasets[x]) for x in ['test']}
    start = time()
    test(model, model_dict[idx].split("/")[-1])
    elapsed = time() - start
    print(elapsed)
