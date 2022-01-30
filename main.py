from __future__ import print_function

import argparse
import time
import numpy as np
import torch

from data import testloader, trainloader
from models import model_factory
from train import Trainer

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning Rate')
parser.add_argument('--steps', '-n', default=200, type=int, help='No of Steps')
parser.add_argument('--gpu', '-p', action='store_true', help='Train on GPU', default='True')
parser.add_argument('--fp16', action='store_true', help='Train with FP16 weights')
parser.add_argument('--loss_scaling', '-s', action='store_true', help='Scale FP16 losses')
parser.add_argument('--model', '-m', default='alexnet2', type=str, help='Name of Network')
args = parser.parse_args()

train_on_gpu = False
if args.gpu and torch.cuda.is_available():
    print("succeed!~!!!!!!!!!")
    train_on_gpu = True
    # CuDNN must be enabled for FP16 training.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

model_name = args.model
model = model_factory(model_name)

trainer = Trainer(model_name, model, args.lr, train_on_gpu, args.fp16,
                  args.loss_scaling)
start = time.time()
trainer.train_and_evaluate(trainloader, testloader, args.steps)
end = time.time()
total = end - start
print(f"total time: {end}")
