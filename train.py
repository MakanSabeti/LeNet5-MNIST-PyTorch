from model import Model
import numpy as np
import torch
import os
import cv2
import sklearn
from sklearn.model_selection import train_test_split
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

if __name__ == '__main__':
    batch_size = 256
    
    current_dir = os.getcwd()
    X, y = [], []
    for i in range(10):
        for d in os.listdir(current_dir + "/assets/{}".format(i)):
            t_img = cv2.imread(current_dir + "/assets/{}".format(i)+"/"+d)
            t_img = cv2.cvtColor(t_img,cv2.COLOR_BGR2GRAY)
            X.append(t_img)
            y.append(i)
    
    X = np.array(X) # X.shape = (6299, 28, 28)
    y = np.array(y) # y.shape = (6299,)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 21)
    y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)
    
    # X_train.shape = (5039, 28, 28)
    # X_test.shape = (1260, 28, 28)
    # y_train.shape = (5039,)
    # y_test.shape = (1260,)
    
    # train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor())
    # test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
    train_loader = DataLoader(X_train, batch_size=batch_size)
    test_loader = DataLoader(X_test, batch_size=batch_size)
    model = Model()
    sgd = SGD(model.parameters(), lr=1e-1)
    loss_fn = CrossEntropyLoss()
    all_epoch = 100

    for current_epoch in range(all_epoch):
        model.train()
        for (idx, train_x) in enumerate(train_loader):
            sgd.zero_grad()
            predict_y = model(train_x.float())
            print(predict_y.shape)
            loss = loss_fn(predict_y, y_train)
            if idx % 10 == 0:
                print('idx: {}, loss: {}'.format(idx, loss.sum().item()))
            loss.backward()
            sgd.step()

        all_correct_num = 0
        all_sample_num = 0
        model.eval()
        for idx, (test_x, test_label) in enumerate(test_loader):
            predict_y = model(test_x.float()).detach()
            predict_y = np.argmax(predict_y, axis=-1)
            current_correct_num = predict_y == test_label
            all_correct_num += np.sum(current_correct_num.numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        acc = all_correct_num / all_sample_num
        print('accuracy: {:.2f}'.format(acc))
        torch.save(model, 'models/printed_{:.2f}.pkl'.format(acc))
