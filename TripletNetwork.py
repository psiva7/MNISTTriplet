from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda

margin = 1.0

def tripletDist(vects):
    dist='euclidean'
    anchor, positive, negative = vects
    #anchor = K.l2_normalize(anchor,axis=-1)
    #negative = K.l2_normalize(negative,axis=-1)
    #positive = K.l2_normalize(positive,axis=-1)
    positive_distance = K.square(anchor - positive)
    negative_distance = K.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = K.sqrt(K.maximum(K.sum(positive_distance, axis=-1, keepdims=True), K.epsilon()))
        negative_distance = K.sqrt(K.maximum(K.sum(negative_distance, axis=-1, keepdims=True), K.epsilon()))
    elif dist == 'sqeuclidean':
        positive_distance = K.mean(positive_distance, axis=-1, keepdims=True)
        negative_distance = K.mean(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    return loss

def create_triplet_network(input_shape, base_network):
        
    input_a = Input(shape=input_shape)
    input_p = Input(shape=input_shape)
    input_n = Input(shape=input_shape)
    
    processed_a = base_network(input_a)
    processed_p = base_network(input_p)
    processed_n = base_network(input_n)
    
    distance = Lambda(tripletDist)([processed_a, processed_p, processed_n])
        
    model = Model([input_a, input_p, input_n], distance)
    
    return model

def tripletLoss(y_true, y_pred):
    lossType = 'maxplus'
    if lossType == 'maxplus':
        loss = K.maximum(0.0, margin + y_pred)
    elif lossType == 'softplus':
        loss = K.log(1 + K.exp(y_pred))
    return K.mean(K.mean(loss))

def tripletAccuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.cast(y_pred < -margin, 'float32'))