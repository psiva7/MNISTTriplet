from keras.models import Model
from keras.layers import Input, Dense

def create_id_network(input_shape, base_network, num_classes):
    input = Input(shape=input_shape)
    processed = base_network(input)
    x = Dense(num_classes, activation='softmax')(processed)
    model = Model([input], x)
    return model 