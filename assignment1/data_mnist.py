import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from struct import *

def mnist() :
    f_train_image = open(r"C:\Users\LG\Documents\CNN\MNIST\train-images.idx3-ubyte", 'rb')
    f_train_label = open(r"C:\Users\LG\Documents\CNN\MNIST\train-labels.idx1-ubyte", 'rb')
    f_test_image = open(r"C:\Users\LG\Documents\CNN\MNIST\t10k-images.idx3-ubyte", 'rb')
    f_test_label = open(r"C:\Users\LG\Documents\CNN\MNIST\t10k-labels.idx1-ubyte", 'rb')

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    img_train = f_train_image.read(16)
    label_train = f_train_label.read(8)
    img_test = f_test_image.read(16)
    label_test = f_test_label.read(8)

    img = np.zeros((28,28))
    index = 0

    k = 0

    while True:
        img_train = f_train_image.read(784)
        label_train = f_train_label.read(1)

        if not img_train:
            break
        if not label_train:
            break
        index = int(label_train[0])
        img = np.reshape(unpack(len(img_train)*'B', img_train), (28, 28))

        X_train.append(img)
        y_train.append(index)

        k += 1

    print("%d train set is done"%k)

    k = 0

    while True:
        img_test = f_test_image.read(784)
        label_test = f_test_label.read(1)

        if not img_test:
            break
        if not label_test:
            break

        index = int(label_test[0])
        img = np.reshape(unpack(len(img_test)*'B', img_test), (28, 28))

        X_test.append(img)
        y_test.append(index)

        k += 1

    print("%d test set is done"%k)

    f_train_image.close()
    f_train_label.close()
    f_test_image.close()
    f_test_label.close()

    return X_train, y_train, X_test, y_test