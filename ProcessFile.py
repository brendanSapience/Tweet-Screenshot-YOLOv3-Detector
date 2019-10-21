import os
import sys

def get_current_dir():
    return os.path.dirname(os.path.abspath(__file__))


#src_path = os.path.join(get_parent_dir(1),'2_Training','src')
utils_path = os.path.join(get_current_dir(),'Utils')
src_path = os.path.join(get_current_dir(),'src')

sys.path.append(src_path)
sys.path.append(utils_path)

import argparse
from keras_yolo3.yolo import YOLO
from PIL import Image
from timeit import default_timer as timer
from utils import load_extractor_model, load_features, parse_input, detect_object
import test
import utils
import pandas as pd
import numpy as np
from Get_File_Paths import GetFileList
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Set up folder names for default values
data_folder = os.path.join(get_current_dir(),'Data')

image_folder = os.path.join(data_folder,'Source_Images')

image_test_folder = os.path.join(image_folder,'Test_Images')

detection_results_folder = os.path.join(image_folder,'Test_Image_Detection_Results') 
detection_results_file = os.path.join(detection_results_folder, 'Detection_Results.csv')

model_folder =  os.path.join(data_folder,'Model_Weights')

model_weights = os.path.join(model_folder,'trained_weights_final.h5')
model_classes = os.path.join(model_folder,'data_classes.txt')

anchors_path = os.path.join(src_path,'keras_yolo3','model_data','yolo_anchors.txt')

FLAGS = None

if __name__ == '__main__':

    FLAGS.model_path = model_weights
    FLAGS.anchors_path = anchors_path
    FLAGS.classes_path = model_classes
    FLAGS.score = 0.25
    FLAGS.box = detection_results_file
    FLAGS.postfix = '_withBox'
    FLAGS.gpu = 1
    FLAGS.no_save_img = False
    FLAGS.output = detection_results_folder
    FLAGS.input_images = image_test_folder

    save_img = not FLAGS.no_save_img

    input_image_paths = GetFileList(FLAGS.input_images)

    print('Found {} input images: {}...'.format(len(input_image_paths), [ os.path.basename(f) for f in input_image_paths[:5]]))

    output_path = FLAGS.output
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # define YOLO detector
    yolo = YOLO(**{"model_path": FLAGS.model_path,
                "anchors_path": FLAGS.anchors_path,
                "classes_path": FLAGS.classes_path,
                "score" : FLAGS.score,
                "gpu_num" : FLAGS.gpu_num,
                "model_image_size" : (416, 416),
                }
               )

    # Make a dataframe for the prediction outputs
    out_df = pd.DataFrame(columns=['image', 'image_path','xmin', 'ymin', 'xmax', 'ymax', 'label','confidence','x_size','y_size'])

    # labels to draw on images
    class_file = open(FLAGS.classes_path, 'r')
    input_labels = [line.rstrip('\n') for line in class_file.readlines()]
    print('Found {} input labels: {}...'.format(len(input_labels), input_labels))

    start = timer()
    text_out = ''

    for i, img_path in enumerate(input_image_paths):
        print(img_path)
        prediction, image = detect_object(yolo, img_path, save_img = save_img,
                                          save_img_path = FLAGS.output,
                                          postfix=FLAGS.postfix)
        y_size,x_size,_ = np.array(image).shape
        for single_prediction in prediction:
            out_df=out_df.append(pd.DataFrame([[os.path.basename(img_path.rstrip('\n')),img_path.rstrip('\n')]+single_prediction + [x_size,y_size]],columns=['image','image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label','confidence','x_size','y_size']))
    end = timer()
    print('Processed {} images in {:.1f}sec - {:.1f}FPS'.format(
         len(input_image_paths), end-start, len(input_image_paths)/(end-start)
         ))
    out_df.to_csv(FLAGS.box,index=False)
