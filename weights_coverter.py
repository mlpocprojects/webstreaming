import os
import tensorflow as tf
import numpy as np
from resources.models.yolov3_tiny import YoloV3Tiny
from resources.models.utils import load_darknet_weights


# TODO:  download weights from url
"""
check if weights folder is available or not
if not create it  and store the download weights in to that file
"""


base_dir = os.getcwd()
initial_weights_file_path = os.path.join(base_dir, 'data/yolo_tiny/weights/yolov3-tiny.weights')
converted_weights_file_path = os.path.join(base_dir, 'data/yolo_tiny/weights/yolov3-tiny.tf')


def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    yolo = YoloV3Tiny(classes=80)
    yolo.summary()
    print('model created')

    load_darknet_weights(yolo, initial_weights_file_path)
    print('weights loaded')

    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    output = yolo(img)
    print('sanity check passed')

    yolo.save_weights(converted_weights_file_path)
    print('weights saved')


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass