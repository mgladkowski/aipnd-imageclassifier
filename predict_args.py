#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Udacity Nanodegree Project
# Image Classifier for AI Programming with Python ND
#
# PROGRAMMER: Michael Gladkowski
# CREATED: 2019-12-12
# REVISED:
# PURPOSE: Retrieve command line arguments for prediction module predict.py.
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
##

import argparse

def get_input_args():
    '''
        Retrieve command line arguments
        
        Returns:
         <argparse.Namespace> : class containing arguments as variables
         
    '''
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="AI Nanodegree Image Classifier Project",
        epilog="""
    Examples:
       python predict.py uploads/daffodil.jpg checkpoint_vgg16.pth\n
       python predict.py uploads/daffodil.jpg checkpoint_vgg16.pth -c cat_to_name.json\n
       python predict.py uploads/daffodil.jpg checkpoint_vgg16.pth --gpu -k 3"""
    )

    # positional
    
    parser.add_argument('image_path', type=str,
                        help='Path to an image or folder of images to predict')
    
    parser.add_argument('checkpoint', type=str,
                        help='Path to checkpoint model')
    
    # switches
    
    parser.add_argument('--gpu', action='store_true', 
                        help='Use GPU if CUDA available')

    # options
    
    parser.add_argument('-c','--category_names', type=str, default='', metavar='file',
                        help='Path to JSON category to names dictionary')
    
    parser.add_argument('-k','--top_k', type=int, default='1',  metavar='K',
                        help='Output top K predictions')

    parser.add_argument('--log', type=str, default='',  metavar='file',
                        help='Filename to save screen output')

    return parser.parse_args()
