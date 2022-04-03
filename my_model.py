import numpy as np
from keras import *
from keras.layers import *
SHAPE_IN = (320,320,3)
HIDDEN = 3

def get_model():  
    return Sequential([
        MaxPooling2D(input_shape=(SHAPE_IN)),
        Conv2D(32,  (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(64,  (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(512, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(), 
        Dense(HIDDEN),                       
    ])
  
    
model = get_encoder_min()
model.load_weights('weights.h5')

#save model
from keras2cpp import export_model
export_model(model, 'my.model')