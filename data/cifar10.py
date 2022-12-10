import torch 
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


def main():
    batchsz = 32

    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor
    ]), download=True)
    cifar_train = DataLoader(cifar_train,batch_size=batchsz,shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor
    ]), download=True)
    cifar_teat = DataLoader(cifar_train,batch_size=batchsz,shuffle=True)

    x, label = iter(cifar_train).next()
    print('x:', x.shape, 'label:', label.shape)

import cv2
import numpy as np
from six.moves import cPickle as pickle
#解压缩二进制文件

def unpack(file):
    fo = open(file, "rb")
    dict = pickle.load(fo,encoding='iso-8859-1')
    fo.close()
    return dict

def test(): 
    testXtr = unpack('cifar10/cifar-10-batches-py/test_batch')
    for i in range(0,10000):
        img = np.reshape(testXtr['data'][i],(3,32,32))
        img = img.transpose(1,2,0)
        img_name = 'cifar10/cifar_test/'  + str(i).zfill(5) + '.jpg'
        cv2.imwrite(img_name, img)


if __name__ == "__main__":
    test()