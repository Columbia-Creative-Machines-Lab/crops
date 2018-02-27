from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import time
import copy
import os
from shutil import copyfile
import argparse
from tqdm import tqdm, trange

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def train_model(model, loss_function, optimizer, num_epochs, save_path, file_name):
    start_time = time.time()
    best_accuracy = 0.0

    for epoch in trange(num_epochs, desc="Epoch"):

        # training phase
        model.train(True)
        running_loss = 0.0
        running_accuracy = 0

        for idx, data in enumerate(tqdm(dataset_loaders['train'], desc="Iter")):
            inputs, labels = data

            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()

            output = model(inputs)
            _, predictions = torch.max(output.data, 1)
            loss = loss_function(output, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            this_batch_num_acc = torch.sum(predictions == labels.data)
            running_accuracy += this_batch_num_acc

            if idx % 200 == 0:
                tqdm.write("[{}, {}], {}".format(epoch, idx, this_batch_num_acc))

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_accuracy = running_accuracy / dataset_sizes['train']

        f = open(save_path+file_name+'_acc.txt', 'a+')
        f.write(str(epoch_accuracy) + "\n")
        f.close()
        f = open(save_path+file_name+'_loss.txt', 'a+')
        f.write(str(epoch_loss) + "\n")
        f.close()

        print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_accuracy))

        # validation phase
        model.train(False)
        running_loss = 0.0
        running_accuracy = 0

        for idx, data in enumerate(tqdm(dataset_loaders['val'], desc="Val")):
            inputs, labels = data

            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()

            output = model(inputs)
            _, predictions = torch.max(output.data, 1)
            loss = loss_function(output, labels)

            running_loss += loss.data[0]
            this_batch_num_acc = torch.sum(predictions == labels.data)
            running_accuracy += this_batch_num_acc

        epoch_loss = running_loss / dataset_sizes['val']
        epoch_accuracy = running_accuracy / dataset_sizes['val']

        f = open(save_path + file_name + '_val_loss.txt', 'a+')
        f.write(str(epoch_loss) + "\n")
        f.close()
        f = open(save_path + file_name + '_val_acc.txt', 'a+')
        f.write(str(epoch_accuracy) + "\n")
        f.close()

        print('Validation Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_accuracy))

        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), save_path + file_name + 'model')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet34")
    parser.add_argument("--opt", type=str, default="sgd")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    data_dir = "./new_data"

    torch.cuda.manual_seed(100)

    datasets = {x: ImageFolder(os.path.join(data_dir, x), data_transforms[x])
         for x in ['train', 'val']}

    dataset_loaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=80,
                    shuffle=True, num_workers=4) for x in ['train', 'val']}

    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

    model = models.resnet34(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    criterion = nn.CrossEntropyLoss()

    if args.opt is "sgd":
        opt = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    elif args.opt is "adam":
        opt = optim.Adam(model.fc.parameters(), lr=3e-4)

    file_name = os.path.basename(__file__)

    start_time = time.ctime().replace(" ", "")
    save_path = './models/' + start_time + '/'
    os.makedirs(save_path)
    full_file_path = os.path.realpath(__file__)
    copyfile(full_file_path, save_path + '/' + file_name)

    model = model.cuda()

    tqdm.write("Training - Base Layers Frozen:")

    train_model(model=model, loss_function=criterion, optimizer=opt,
        num_epochs=1, save_path=save_path, file_name=file_name)

    for param in model.parameters():
        param.requires_grad = True

    if args.opt is "sgd":
        opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif args.opt is "adam":
        opt = optim.Adam(model.parameters(), lr=3e-4)

    tqdm.write("Training:")
     
    train_model(model=model, loss_function=criterion, 
        optimizer=opt, num_epochs=args.epochs,
        save_path=save_path, file_name=file_name)
