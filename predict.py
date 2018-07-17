#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json
import tensorflow as tf
import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')

def _main_(args):
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################    

    yolo.load_weights(weights_path)



    ###############################
    #   Predict bounding boxes 
    ###############################

    if image_path[-4:] == '.mp4':
        video_out = image_path[:-4] + '_detected_full_yolo' + image_path[-4:]
        video_reader = cv2.VideoCapture(image_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'MPEG'),
                               20.0, (960, 1080))

        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()
            imageStereo = image[0:1080, 480:1440, :]
            originalSize = (imageStereo.shape[1], imageStereo.shape[0])

            imageStereo = cv2.resize(imageStereo, (1024, 512))
            # imageLeft = image[0:540, 480:1440, :]

            # imageLeft = cv2.resize(imageLeft, (512, 512))
            # imageRight = image[540:1080, 480:1440, :]
            # imageRight = cv2.resize(imageRight, (512, 512))
            #
            # imageStereo = cv2.vconcat([imageLeft, imageRight])
            start = time.time()
            boxes = yolo.predict(imageStereo)
            end = time.time()
            print(end - start)
            imageStereo = draw_boxes(imageStereo, boxes, config['model']['labels'])

            #boxes = yolo.predict(imageRight)
            #imageRight = draw_boxes(imageRight, boxes, config['model']['labels'])

            imageStereo = cv2.resize(imageStereo, originalSize)
            #imageRight = cv2.resize(imageRight, originalSiye)

            # cv2.imshow("predicted left", imageStereo)
            #cv2.imshow("predicted right", imageRight)
            cv2.waitKey(1)
            video_writer.write(np.uint8(imageStereo))

        video_reader.release()
        video_writer.release()
    else:
        image = cv2.imread(image_path)
        boxes = yolo.predict(image)
        image = draw_boxes(image, boxes, config['model']['labels'])

        print(len(boxes), 'boxes are found')

        cv2.imwrite(image_path[:-4] + '_detected_full_yolo' + image_path[-4:], image)

if __name__ == '__main__':
    args = argparser.parse_args(['-c', 'config.json', '-w', 'best_saved_weights_full_yolo.h5', '-i', 'Test/test5.mp4'])
    _main_(args)
