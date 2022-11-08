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


mnist_99 = open(os.path.join((os.getcwd() +'\models\mnist_0.99.pkl')), encoding="utf8")
data = pickle.load(mnist_99)

# test = pickle.load(open('models\mnist_0.99.pkl','rb'))
