#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Udacity Nanodegree Project
# Image Classifier for AI Programming with Python ND
#
# PROGRAMMER: Michael Gladkowski
# CREATED: 2019-12-12
# REVISED:
# PURPOSE: Client-side classifier.  Loads a checkpoint created by myclassifier.py 
#          and performs image classification.  Human names can be provided via 
#          options below, otherwise only classes are displayed.  Input path can be
#          a single image or a folder of images.
#          
# Arguments where () indicated default value:
#
#       Positional:
#           image_path            Path to an image or folder of images to predict
#           checkpoint            Path to checkpoint model
#
#       Switches:
#           --gpu                 Use GPU if CUDA available
#
#       Options:
#           -c,--category_names   Path to JSON category to names dictionary
#           -k,--top_k            Output top K predictions (1)
#           --log                 Filename to save screen output
#
#   Example call:
#
#    python predict.py uploads/sunflower.jpg models/checkpoint.pth -c cat_to_name.json --gpu -k 3
#
##

# import modules

from os import listdir,path
from datetime import datetime

# program modules

import output
import predict_args
from myclassifier import MyClassifier


def main():
    '''
        MAIN PROGRAM
        
    '''
    # get command line arguments
    
    args = predict_args.get_input_args()
    
    # display header
    
    output.logfile = args.log
    output.log("Image Classifier Project for AI Programming with Python Nanodegree")
    output.log()
    
    # model
    
    classifier = MyClassifier()
    
    if args.category_names:
        classifier.data.load_names(args.category_names)
    
    
    # set compute device
    
    classifier.set_device(args.gpu)

    # load checkpoint
    
    classifier.load_checkpoint(args.checkpoint, training=False)


    # turn path into iterator to allow multiple files
    
    images = []
    
    if path.isdir(args.image_path):
        images = [path.join(args.image_path,f) 
                  for f in listdir(args.image_path) 
                  if path.isfile(path.join(args.image_path, f))]
        
    elif path.isfile(args.image_path):
        images = [args.image_path]
        
    
    # run predictions
    
    for image in images:
        
        classifier.predict(image, args.top_k)
    
    


# RUN PROGRAM
#
if __name__ == "__main__":
    main()
    