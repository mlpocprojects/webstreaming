from .yolov3_tiny import YoloV3Tiny
import os

base_dir = os.getcwd()
weights_dir = os.path.join(base_dir, 'data/yolo_tiny/weights/yolov3-tiny.tf')
classes_dir = os.path.join(base_dir, 'data/yolo_tiny/classes/coco.names')


def load_pre_trained_model_classes():
    if os.path.isfile(classes_dir):
        yolo = YoloV3Tiny(classes=80)
        yolo.load_weights(weights_dir)
        print('weights loaded')
        class_names = [c.strip() for c in open(classes_dir).readlines()]
        print('classes loaded')
        return class_names, yolo
    else:
        return 'Model file is not present in the desired location. ' \
               'please check the if you have the /model/weights dir'
