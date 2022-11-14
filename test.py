import numpy as np
import torch
import os
import cv2
import sklearn
import pickle
from sklearn.model_selection import train_test_split
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

with open('models/mnist_0.98', "rb") as payload_file:
    model = pickle.load(payload_file)