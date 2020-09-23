#! /usr/bin/env python3

"""
@authors: Brian Hutchinson (Brian.Hutchinson@wwu.edu)
          Michael Albert (albertm4@wwu.edu)
          Archan Rupela (rupelaa@wwu.edu)

A simple example of building a convolutional neural network using
PyTorch.

For usage, run with the -h flag.

Disclaimers:
- Distributed as-is.
- Please contact me if you find any issues with the code.

"""

import torch
import torch.nn.functional as F
import argparse
import sys
import numpy as np
import pdb

class MNISTDataset():
    def __init__(self, input, labels):
        """
        In the constructor we create a map-style dataset using the paths to the input features and targets
        """
        temp = np.load(input).astype(np.float32)
        self.inputs = temp.reshape(len(temp),1,28,28)
        self.targets = np.load(labels).astype(np.int64)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)

    def D(self):
        return self.inputs.shape[0]

class SimpleConvNeuralNet(torch.nn.Module):
    def __init__(self, C):
        """
        In the constructor we create a convolutional model with 2 convolutions,
        a max pool, and 2 fully connected layers.
        """
        super(SimpleConvNeuralNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,8,3)
        self.conv2 = torch.nn.Conv2d(8,16,3)
        self.pool = torch.nn.MaxPool2d(2,stride=2)
        self.fc1 = torch.nn.Linear(16*12*12,128)
        self.fc2 = torch.nn.Linear(128,C)

        # Print params
        for name, param in self.named_parameters():
            print(name,param.data.shape)

    def forward(self, x, f1):
        """
        In the forward function we accept a Tensor of input data and we must
        return a Tensor of output data. We can use Modules defined in the
        constructor as well as arbitrary operators on Tensors.
        """
        if f1 == "relu":
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(-1, 16*12*12)
            x = self.fc1(x)
            x = self.fc2(x)
        elif f1 == "tanh":
            x = F.tanh(self.conv1(x))
            x = F.tanh(self.conv2(x))
            x = self.pool(x)
            x = x.view(-1, 16*12*12)
            x = self.fc1(x)
            x = self.fc2(x)
        elif f1 == "sigmoid":
            x = F.logsigmoid(self.conv1(x))
            x = F.logsigmoid(self.conv2(x))
            x = x.view(-1, 16*12*12)
            x = self.pool(x)
            x = self.fc1(x)
            x = self.fc2(x)
        return x

class OurConvNeuralNet(torch.nn.Module):
    def __init__(self, C):
        """
        In the constructor we create a convolutional model with 2 convolutions,
        a max pool, and 2 fully connected layers.
        """
        super(OurConvNeuralNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,32,5)
        self.conv2 = torch.nn.Conv2d(32,64,5)
        self.pool = torch.nn.MaxPool2d(2,stride=2)
        self.fc1 = torch.nn.Linear(64*4*4,128)
        self.fc2 = torch.nn.Linear(128,C)

        # Print params
        for name, param in self.named_parameters():
            print(name,param.data.shape)

    def forward(self, x, f1):
        """
        In the forward function we accept a Tensor of input data and we must
        return a Tensor of output data. We can use Modules defined in the
        constructor as well as arbitrary operators on Tensors.
        """
        if f1 == "relu":
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(-1, 64*4*4)
            x = self.fc1(x)
            x = self.fc2(x)
        elif f1 == "tanh":
            x = F.tanh(self.conv1(x))
            x = self.pool(x)
            x = F.tanh(self.conv2(x))
            x = self.pool(x)
            x = x.view(-1, 64*4*4)
            x = self.fc1(x)
            x = self.fc2(x)
        elif f1 == "sigmoid":
            x = F.logsigmoid(self.conv1(x))
            x = self.pool(x)
            x = F.logsigmoid(self.conv2(x))
            x = self.pool(x)
            x = x.view(-1, 64*4*4)
            x = self.fc1(x)
            x = self.fc2(x)
        return x

def parse_all_args():
    # Parses commandline arguments

    parser = argparse.ArgumentParser()

    parser.add_argument("C",help="The number of classes if classification or output dimension if regression (int)",type=int)
    parser.add_argument("train_x",help="The training set input data (npz)")
    parser.add_argument("train_y",help="The training set target data (npz)")
    parser.add_argument("dev_x",help="The development set input data (npz)")
    parser.add_argument("dev_y",help="The development set target data (npz)")

    parser.add_argument("-f1",type=str,\
            help="The hidden activation function: \"relu\" or \"tanh\" or \"sigmoid\" (string) [default: \"relu\"]",default="relu")
    parser.add_argument("-opt",type=str,\
            help="The optimizer: \"adadelta\", \"adagrad\", \"adam\",\"rmsprop\", \"sgd\" (string) [default: \"adam\"]",default="adam")
    parser.add_argument("-lr",type=float,\
            help="The learning rate (float) [default: 0.1]",default=0.1)
    parser.add_argument("-mb",type=int,\
            help="The minibatch size (int) [default: 32]",default=32)
    parser.add_argument("-report_freq",type=int,\
            help="Dev performance is reported every report_freq updates (int) [default: 128]",default=128)
    parser.add_argument("-epochs",type=int,\
            help="The number of training epochs (int) [default: 100]",default=100)

    return parser.parse_args()

def train(model,train_loader,dev_loader,args):
    # determine device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = None
    if optimizer == None:
        if args.opt == "adadelta":
            optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
        if args.opt == "adagrad":
            optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
        if args.opt == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.opt == "rmsprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
        if args.opt == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):

        for update,(mb_x,mb_y) in enumerate(train_loader):
            mb_x = mb_x.to(device)
            mb_y = mb_y.to(device)
            mb_y_pred = model(mb_x, args.f1) # evaluate model forward function
            loss      = criterion(mb_y_pred,mb_y) # compute loss

            optimizer.zero_grad() # reset the gradient values
            loss.backward()       # compute the gradient values
            optimizer.step()      # apply gradients

            if (update % args.report_freq) == 0:
                # eval on dev once per epoch
                dev_acc = 0
                devN = 0
                for _,(mb_x,mb_y) in enumerate(dev_loader):
                    mb_x = mb_x.to(device)
                    mb_y = mb_y.to(device)
                    mb_y_pred     = model(mb_x, args.f1)
                    _,mb_y_pred_i = torch.max(mb_y_pred,1)
                    dev_acc          += ((mb_y_pred_i == mb_y).sum()).item()
                    devN            += len(mb_x)
                dev_acc = dev_acc/devN
                print("%03d.%04d: dev %.3f" % (epoch,update,dev_acc))

def main(argv):
    # determine device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # parse arguments
    args = parse_all_args()

    # load data
    train_data = MNISTDataset(args.train_x, args.train_y)
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=args.mb, shuffle=True, drop_last=False)
    dev_data = MNISTDataset(args.dev_x, args.dev_y)
    dev_loader = torch.utils.data.DataLoader(dev_data,batch_size=args.mb, shuffle=False, drop_last=False)

    #model = SimpleConvNeuralNet(args.C) # original CNN
    model = OurConvNeuralNet(args.C) # new CNN
    model = model.to(device)
    train(model,train_loader,dev_loader,args)
    torch.save(model, "model.pt")

if __name__ == "__main__":
    main(sys.argv)
