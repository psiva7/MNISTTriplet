

import keras
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import plot_model

from MNISTGenerator import DataGenerator

from ViewMNIST import GetClrImgs
from ViewMNIST import PlotResult

from TripletNetwork import create_triplet_network
from TripletNetwork import tripletLoss
from TripletNetwork import tripletAccuracy

from IDNetwork import create_id_network

from MNISTHelpers import get_reid_test_data, get_train_data, get_num_classes

from ReIDHelpers import GetRank1Accuracy
from ReIDHelpers import GetFeatMatrix

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
x_train, y_train = get_train_data()
x_trainClr = GetClrImgs(x_train, y_train)

probe, pLabel, gallery, gLabel = get_reid_test_data()
galleryClr = GetClrImgs(gallery, gLabel)

input_shape = x_train.shape[1:]

'''
# ID network test
base_network_for_ID = create_base_network(input_shape)

IDModel = create_id_network(input_shape, base_network_for_ID, num_classes)
#plot_model(IDModel, to_file='IDNetwork.png', show_shapes='True')

epochs = 10
learningRate = 0.002
beta1 = 0.9
beta2 = 0.999
IDModel.compile(loss=keras.losses.categorical_crossentropy, optimizer=Adam(lr=learningRate, beta_1=beta1, beta_2=beta2), metrics=['accuracy'])
one_hot_labels = keras.utils.to_categorical(y_train, num_classes=10)
IDModel.fit(x=x_train, y=one_hot_labels, batch_size=100, epochs=epochs)

tripletRank1 = GetRank1Accuracy(probe, pLabel, gallery, gLabel, galleryClr, base_network_for_ID, saveIdx=-1, saveFigName="ID.png", normalize = False)
tripletNormRank1 = GetRank1Accuracy(probe, pLabel, gallery, gLabel, galleryClr, base_network_for_ID, saveIdx=-1, saveFigName="IDNorm.png", normalize = True)

trainFeat = GetFeatMatrix(x_train, base_network_for_ID, -1, False)
PlotResult(x_trainClr, trainFeat, False, "IDTrain.png")

print('\n\n' + str(tripletRank1) + ',' + str(tripletNormRank1))
with open("result.txt", "a") as myfile:
    myfile.write('ID,' + str(tripletRank1) + ',' + str(tripletNormRank1) + '\n')
'''

#************
#************
# Triplet test by setting useSemiHardPos and useSemiHardNeg to false in DataGenerator.
# This will be used as initialization network for semi-hard triplets later.
# This should in theory give exactly the same result as setting useSemiHardPos=False, useSemiHardNeg=False and topPCT=1.0 
#************
#************

# Triplet Network with random triplets
base_network = create_base_network(input_shape)

modelRandTriplet = create_triplet_network(input_shape, base_network)
#plot_model(modelRandTriplet, to_file='RandTripletNetwork.png', show_shapes='True')

epochs = 3
learningRate = 0.002
beta1 = 0.9
beta2 = 0.999
modelRandTriplet.compile(loss=tripletLoss, optimizer=Adam(lr=learningRate, beta_1=beta1, beta_2=beta2), metrics=[tripletAccuracy])

bSize = 100
training_generator = DataGenerator(x_train, y_train, num_classes, base_network, bSize, False, False)
modelRandTriplet.fit_generator(generator=training_generator,
            steps_per_epoch=training_generator.__len__(),
            epochs = epochs,
            use_multiprocessing=False, shuffle=True)

tripletRank1 = GetRank1Accuracy(probe, pLabel, gallery, gLabel, galleryClr, base_network, saveIdx=-1, saveFigName="RandTrip.png", normalize = False)
tripletNormRank1 = GetRank1Accuracy(probe, pLabel, gallery, gLabel, galleryClr, base_network, saveIdx=-1, saveFigName="RandTripNorm.png", normalize = True)

trainFeat = GetFeatMatrix(x_train, base_network, -1, False)
PlotResult(x_trainClr, trainFeat, False, "RandTripTrain.png")

print('\n\n' + str(tripletRank1) + ',' + str(tripletNormRank1))
with open("result.txt", "a") as myfile:
    myfile.write('RandTrip,' + str(tripletRank1) + ',' + str(tripletNormRank1) + '\n')


#************
#************
# We set useSemiHardPos=False, useSemiHardNeg=False and topPCT=1.0 tos see if this gives similar result to
# settig useSemiHardPos=True and useSemiHardNeg=True (above test)
#************
#************

# Triplet Network with random triplets
base_networkT = create_base_network(input_shape)

modelRandTriplet = create_triplet_network(input_shape, base_networkT)
#plot_model(modelRandTriplet, to_file='RandTripletNetwork.png', show_shapes='True')

epochs = 3
learningRate = 0.002
beta1 = 0.9
beta2 = 0.999
modelRandTriplet.compile(loss=tripletLoss, optimizer=Adam(lr=learningRate, beta_1=beta1, beta_2=beta2), metrics=[tripletAccuracy])

bSize = 100
training_generator = DataGenerator(x_train, y_train, num_classes, base_networkT, bSize, True, True, 1.0)
modelRandTriplet.fit_generator(generator=training_generator,
            steps_per_epoch=training_generator.__len__(),
            epochs = epochs,
            use_multiprocessing=False, shuffle=True)

tripletRank1 = GetRank1Accuracy(probe, pLabel, gallery, gLabel, galleryClr, base_networkT, saveIdx=-1, saveFigName="RandTripT.png", normalize = False)
tripletNormRank1 = GetRank1Accuracy(probe, pLabel, gallery, gLabel, galleryClr, base_networkT, saveIdx=-1, saveFigName="RandTripNormT.png", normalize = True)

trainFeat = GetFeatMatrix(x_train, base_networkT, -1, False)
PlotResult(x_trainClr, trainFeat, False, "RandTripTrainT.png")

print('\n\n' + str(tripletRank1) + ',' + str(tripletNormRank1))
with open("result.txt", "a") as myfile:
    myfile.write('RandTripT,' + str(tripletRank1) + ',' + str(tripletNormRank1) + '\n')



#************
#************
# Using random initalized network we train for 3 epochs using semi-hard triplet mining. Where, semi-hard
# triplets are selected in each batch by sorting distances to all negatives/postives and selecting randomly
# a point in the hardest topPCT. If topPCT<=0 then the hardest sample is selected.
#************
#************

randRangeList = [1.0, 0.75, 0.5, 0.25, -1]

for randRange in randRangeList:
    # Triplet Network with hard triplets
    base_network2 = create_base_network(input_shape)

    for i in range(0,len(base_network.layers)):
        base_network2.layers[i].set_weights(base_network.layers[i].get_weights())

    modelRandTriplet2 = create_triplet_network(input_shape, base_network2)
    #plot_model(modelRandTriplet2, to_file='SemiHardTripletNetwork.png', show_shapes='True')

    epochs = 3
    learningRate = 0.002
    beta1 = 0.9
    beta2 = 0.999
    modelRandTriplet2.compile(loss=tripletLoss, optimizer=Adam(lr=learningRate, beta_1=beta1, beta_2=beta2), metrics=[tripletAccuracy])
    import matplotlib.pyplot as plt
    bSize = 100
    training_generator = DataGenerator(x_train, y_train, num_classes, base_network2, bSize, True, True, randRange)

    modelRandTriplet2.fit_generator(generator=training_generator,
                steps_per_epoch=training_generator.__len__(),
                epochs = epochs,
                use_multiprocessing=False, shuffle=True)

    tripletRank1 = GetRank1Accuracy(probe, pLabel, gallery, gLabel, galleryClr, base_network2, saveIdx=-1, saveFigName="SemiHardTrip%.2f.png" % randRange, normalize = False)
    tripletNormRank1 = GetRank1Accuracy(probe, pLabel, gallery, gLabel, galleryClr, base_network2, saveIdx=-1, saveFigName="SemiHardTripNorm%.2f.png" % randRange, normalize = True)

    trainFeat = GetFeatMatrix(x_train, base_network2, -1, False)
    PlotResult(x_trainClr, trainFeat, False, "SemiHardTripTrain%.2f.png" % randRange)

    print('\n\n' + str(tripletRank1) + ',' + str(tripletNormRank1))
    with open("result.txt", "a") as myfile:
        myfile.write('SemiHardTrip%.2f,' % randRange + str(tripletRank1) + ',' + str(tripletNormRank1) + '\n')
