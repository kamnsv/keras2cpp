import tensorflow as tf
from keras2cpp import export_model

def convert_model(dname):
    
    model = tf.keras.models.load_model(dname)
    export_model(model, f'{dname}.model')
    
if __name__=='__main__':
    import sys
    if len(sys.argv) > 1:
        convert_model(sys.argv[1])
    else:
        print('Pass the path to the model')