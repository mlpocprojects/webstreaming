from keras.applications.vgg16 import VGG16
import os


# handle GPU memory Overloading
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)


def load_pre_trained_model(model_path: str):
    if os.path.isfile(model_path):
        return VGG16(weights=model_path)
    else:
        return 'Model file is not present in the desired location. ' \
               'please check the if you have the /model/weights dir'
