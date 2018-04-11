import numpy as np
import scipy as sp
import scipy.spatial.distance
import matplotlib.pyplot as plt

from keras import backend as K

from ViewMNIST import PlotResult

def GetFeature(x, functor, saveIdx, normalize=False):
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

def GetFeatMatrix(X, model, saveIdx, normalize=False):
    inp = model.get_input_at(0)                                          # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    #outputNames = [layer.name for layer in model.layers]
    functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function
    embedding = GetFeature(X, functor, saveIdx, normalize)
    return embedding

def GetRank1Accuracy(probe, pLabel, gallery, gLabel, galleryClr, base_network, saveIdx, saveFigName=None, normalize = False):
    probeFeat = GetFeatMatrix(probe, base_network, saveIdx, normalize)
    galleryFeat = GetFeatMatrix(gallery, base_network, saveIdx, normalize)
    dist = sp.spatial.distance.cdist(galleryFeat, probeFeat)
    
    TP = 0
    for i in range(0,dist.shape[1]):
        minIdx = np.argmin(dist[:,i])
        if pLabel[i] == gLabel[minIdx]:
            TP += 1
    
    rank1 = (TP/dist.shape[1] * 100)
    if (saveFigName is not None):
        PlotResult(galleryClr, galleryFeat, False, saveFigName)
    
    return rank1