

import keras
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import plot_model

from MNISTGenerator import DataGenerator

from ViewMNIST import GetClrImgs

from TripletNetwork import create_triplet_network
from TripletNetwork import tripletLoss
from TripletNetwork import tripletAccuracy

from IDNetwork import create_id_network

from MNISTHelpers import get_reid_test_data, get_train_data, get_num_classes

from ReIDHelpers import GetRank1Accuracy

num_classes = get_num_classes()


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2)(x)
    return Model(input, x)


# Get the training and testing data
x_train2, y_train2 = get_train_data()

probe, pLabel, gallery, gLabel = get_reid_test_data()
galleryClr = GetClrImgs(gallery, gLabel)

input_shape = x_train2.shape[1:]

# ID network test
base_network_for_ID = create_base_network(input_shape)

IDModel = create_id_network(input_shape, base_network_for_ID, num_classes)
#plot_model(IDModel, to_file='IDNetwork.png', show_shapes='True')

epochs = 10
learningRate = 0.002
beta1 = 0.9
beta2 = 0.999
IDModel.compile(loss=keras.losses.categorical_crossentropy, optimizer=Adam(lr=learningRate, beta_1=beta1, beta_2=beta2), metrics=['accuracy'])
one_hot_labels = keras.utils.to_categorical(y_train2, num_classes=10)
IDModel.fit(x=x_train2, y=one_hot_labels, batch_size=100, epochs=epochs)

tripletRank1 = GetRank1Accuracy(probe, pLabel, gallery, gLabel, galleryClr, base_network_for_ID, saveIdx=-1, saveFigName="V10ID.png", normalize = False)
tripletNormRank1 = GetRank1Accuracy(probe, pLabel, gallery, gLabel, galleryClr, base_network_for_ID, saveIdx=-1, saveFigName="V10IDNorm.png", normalize = True)

print('\n\n' + str(tripletRank1) + ',' + str(tripletNormRank1))
with open("result.txt", "a") as myfile:
    myfile.write('V10ID,' + str(tripletRank1) + ',' + str(tripletNormRank1) + '\n')

# Triplet Network with random triplets
base_network = create_base_network(input_shape)

modelRandTriplet = create_triplet_network(input_shape, base_network)
#plot_model(modelRandTriplet, to_file='RandTripletNetwork.png', show_shapes='True')

epochs = 10
learningRate = 0.002
beta1 = 0.9
beta2 = 0.999
modelRandTriplet.compile(loss=tripletLoss, optimizer=Adam(lr=learningRate, beta_1=beta1, beta_2=beta2), metrics=[tripletAccuracy])

bSize = 100
training_generator = DataGenerator(x_train2, y_train2, num_classes, base_network, bSize, False)
modelRandTriplet.fit_generator(generator=training_generator,
            steps_per_epoch=training_generator.__len__(),
            epochs = epochs,
            use_multiprocessing=False, shuffle=True)

tripletRank1 = GetRank1Accuracy(probe, pLabel, gallery, gLabel, galleryClr, base_network, saveIdx=-1, saveFigName="V10Trip.png", normalize = False)
tripletNormRank1 = GetRank1Accuracy(probe, pLabel, gallery, gLabel, galleryClr, base_network, saveIdx=-1, saveFigName="V10TripNorm.png", normalize = True)

print('\n\n' + str(tripletRank1) + ',' + str(tripletNormRank1))
with open("result.txt", "a") as myfile:
    myfile.write('V10Trip,' + str(tripletRank1) + ',' + str(tripletNormRank1) + '\n')

'''
for i in range(0,len(base_network.layers)):
    base_network2.layers[i].set_weights(base_network.layers[i].get_weights())

epochs = 1

training_generator = DataGenerator(x_train2, y_train2, num_classes, base_network2, bSize, True)
model.fit_generator(generator=training_generator,
            steps_per_epoch=training_generator.__len__(),
            epochs = epochs,
            use_multiprocessing=False, shuffle=True)

tripletRank1 = GetRank1Accuracy(probe, pLabel, gallery, gLabel, galleryClr, base_network, saveIdx=-1, saveFigName="V10TripV2.png", normalize = False)
tripletNormRank1 = GetRank1Accuracy(probe, pLabel, gallery, gLabel, galleryClr, base_network, saveIdx=-1, saveFigName="V10TripNormV2.png", normalize = True)

print('\n\n' + str(tripletRank1) + ',' + str(tripletNormRank1))
with open("result.txt", "a") as myfile:
    myfile.write('V10TripV2,' + str(tripletRank1) + ',' + str(tripletNormRank1) + '\n')
'''