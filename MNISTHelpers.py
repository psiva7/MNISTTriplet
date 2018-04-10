import numpy as np
from keras.datasets import mnist

def create_gallery_probe(x, digit_indices, num_classes):
    probe = []
    probe_l = []
    gallery = []
    gallery_l = []
    n = min([len(digit_indices[d]) for d in range(num_classes)])
    numProbe = max(int(n*0.25),1)
    for d in range(num_classes):
        for i in range(n):
            z1 = digit_indices[d][i]
            if i < numProbe:
                probe += [[x[z1]]]
                probe_l.append(d)
            else:
                gallery += [[x[z1]]]
                gallery_l.append(d)
    return np.array(probe), np.array(probe_l), np.array(gallery), np.array(gallery_l)

def get_num_classes():
    return 10

def get_train_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_train /= 255
    x_train -= 0.5
    x_train *= 2

    num_classes = get_num_classes()

    locs = dict()
    minNum = 0
    for i in range(0,num_classes):
        locs[i] = np.where(y_train==i)[0]
        if i == 0:
            minNum = locs[i].shape[0]
        elif (minNum > locs[i].shape[0]):
            minNum = locs[i].shape[0]

    # return data such that every set of 10 data points has a sample from each class
    x_train2 = np.zeros((minNum//num_classes * num_classes * num_classes, x_train.shape[1], x_train.shape[2]), np.float)
    y_train2 = np.zeros((minNum//num_classes * num_classes * num_classes),np.uint8)
    idx = -1
    idx2 = -1
    for i in range(0,minNum//num_classes):
        for j in range(0,num_classes):
            idx2 = idx2 + 1
            for cls in range(0,num_classes):
                idx = idx + 1
                oriIdx = locs[cls][idx2]
                x_train2[idx,:,:] = x_train[oriIdx,:,:]
                y_train2[idx] = y_train[oriIdx]
    return x_train2, y_train2

def get_reid_test_data():
    num_classes = get_num_classes()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype('float32')
    x_test /= 255
    x_test -= 0.5
    x_test *= 2
    digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    [probe, pLabel, gallery, gLabel] = create_gallery_probe(x_test, digit_indices, num_classes)
    probe = probe[:,0]
    gallery = gallery[:,0]
    return probe, pLabel, gallery, gLabel