import numpy as np
import scipy as sp
import keras
import random
from keras import backend as K
import tensorflow as tf


# Modified from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x_train, y_train, num_classes, base_network, batch_size=100, useSemiHardPos = False, useSemiHardNeg = False, topPCT = 1.0):
        'Initialization'
        '''
        Generator assumes: 
        - batch_size is a multiple of num_classes
        - x_train and y_train are sorted such that the data are class0, class1, ... classN, class0, ... so on
          where N+1 = num_classes and y_train.shape[0] is a multiple of num_classes. Refer to MNISTHelpers get_train_data()
          for example of generating x_train and y_train
        '''
        self.batch_size = batch_size
        self.x_train = x_train
        self.y_train = y_train
        self.num_classes = num_classes
        self.base_network = base_network
        self.graph = tf.get_default_graph()
        self.useSemiHardPos = useSemiHardPos
        self.useSemiHardNeg = useSemiHardNeg
        self.topPCT = topPCT

    def __len__(self):
        'Denotes the number of batches per epoch'
        len = int(np.floor(self.x_train.shape[0] / self.batch_size))-1
        return len

    def __getitem__(self, index):
        'Generate one batch of data'
        startIdx = self.batch_size*index
        endIdx = startIdx + self.batch_size

        curX = self.x_train[startIdx:endIdx,:,:]
        curY = self.y_train[startIdx:endIdx]

        if (self.useSemiHardPos or self.useSemiHardNeg):
            with self.graph.as_default():
                curX_Feat = self.GetFeatMatrix(curX, self.base_network, -1, normalize=False)
            pairDists = sp.spatial.distance.pdist(curX_Feat)
            pairDistMat = sp.spatial.distance.squareform(pairDists)
        
        if self.useSemiHardPos:
            posX = self.GetHard(curX, curX_Feat, pairDistMat, curY, 'pos')
        else:
            posX = self.GetRand(curX, curY, 'pos')
        
        if self.useSemiHardNeg:
            negX = self.GetHard(curX, curX_Feat, pairDistMat, curY, 'neg')
        else:
            negX = self.GetRand(curX, curY, 'neg')
        
        X = [curX, posX, negX]
        y = curY
        return X, y
    
    def GetRand(self, curX, curY, pairType):
        idx = []
        for i in range(0,curX.shape[0]):
            srcClsIdx = curY[i]
            validIdx = []
            for j in range(0, curX.shape[0]):
                curClsIdx = curY[j]
                if i == j:
                    continue
                if pairType == 'pos' and srcClsIdx != curClsIdx:
                    continue
                if pairType == 'neg' and srcClsIdx == curClsIdx:
                    continue
                validIdx.append(j)
            np.random.shuffle(validIdx)
            idx.append(validIdx[0])
        Imgs = curX[idx,:,:]
        return Imgs
    
    def GetHard(self, curX, curX_Feat, pairDistMat, curY, pairType):
        idx = []
        for i in range(0,curX.shape[0]):
            srcClsIdx = curY[i]
            distList = []
            distIdxList = []
            for j in range(0, curX.shape[0]):
                curClsIdx = curY[j]
                if i == j:
                    continue
                if pairType == 'pos' and srcClsIdx != curClsIdx:
                    continue
                if pairType == 'neg' and srcClsIdx == curClsIdx:
                    continue
                dist = pairDistMat[i,j]
                distList.append(dist)
                distIdxList.append(j)
            distList = np.array(distList)
            distIdxList = np.array(distIdxList)
            locs = np.argsort(distList)
            if (self.topPCT <= 0 or self.topPCT > 1.0):
                randIdx = 0
            else:
                upperBound = max(1,int(len(locs)*self.topPCT+0.5))
                randIdx = np.random.randint(0, upperBound)
            if pairType == 'pos':
                idx.append(distIdxList[locs[-(randIdx+1)]])
            else:
                idx.append(distIdxList[locs[randIdx]])
        posImg = curX[idx,:,:]
        return posImg
    
    def GetFeature(self, x, functor, saveIdx, normalize=False):
        # Duplicates code in ReIDHelpers
        embedding = None
        try:
            layer_outs = functor([x, 0.])
            embedding = layer_outs[saveIdx]
            if (normalize == True):
                norm = np.sqrt(embedding[:,0]*embedding[:,0] + embedding[:,1]*embedding[:,1])
                norm[norm == 0] = np.finfo(float).eps
                embedding[:,0] = embedding[:,0] / norm
                embedding[:,1] = embedding[:,1] / norm
        except OSError:
            print('Feat error')
        
        return embedding

    def GetFeatMatrix(self, X, model, saveIdx, normalize=False):
        # Duplicates code in ReIDHelpers
        inp = model.get_input_at(0)                                          # input placeholder
        outputs = [layer.output for layer in model.layers]          # all layer outputs
        #outputNames = [layer.name for layer in model.layers]
        functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function
        embedding = self.GetFeature(X, functor, saveIdx, normalize)
        return embedding
    
    def on_epoch_end(self):
        '''
        Shuffle in chunks of num_classes so that a batch of size > num_classes
        will have different set of data. Note we assume batch size is a 
        multiple of num_classes
        '''
        ind = np.arange(self.y_train.shape[0])
        ind = ind.reshape(-1,self.num_classes)
        np.random.shuffle(ind)
        ind=ind.flatten()
        self.y_train = self.y_train[ind]
        self.x_train = self.x_train[ind]

    def create_triplet(self):
        '''
        Currently Not used:
        Positive and negative pair creation.
        Alternates between positive and negative pairs.
        '''
        num_classes = 10
        digit_indices = [np.where(self.y_train == i)[0] for i in range(num_classes)]
        triplets = []
        label = []
        n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
        for d in range(num_classes):
            for i in range(n):
                z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
                inc = random.randrange(1, num_classes)
                dn = (d + inc) % num_classes
                z3 = digit_indices[dn][i]
                triplets += [[self.x_train[z1], self.x_train[z2], self.x_train[z3]]]
                label.append(1)
        return np.array(triplets), label