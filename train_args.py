#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Udacity Nanodegree Project
# Image Classifier for AI Programming with Python ND
#
# PROGRAMMER: Michael Gladkowski
# CREATED: 2019-12-12
# REVISED:
# PURPOSE: Retrieve command line arguments for the training module train.py.
#
# Arguments where () indicated default value:
#
#       Positional:
#           data_dir              Folder of training images in PyTorch dataloader structure
#
#       Switches:
#           --gpu                 Use GPU to train the model if CUDA available
#           --save_optimizer      Include optimizer state_dict in checkpoint save
#
#       Options:
#           --arch                Pre-trained model to use : alexnet|resnet152|(vgg16)
#           --load                Filename of checkpoint to load and resume training
#           --log                 Filename to copy screen output to
#           --save                Filename of checkpoint to save after training, or, 
#           --save_dir            Folder in which to save auto-named checkpoints after training
#           --save_accuracy       Start saving best checkpoints after this accuracy is reached
#           --hidden_units        List of hidden layer sizes ([1024])
#           -e, --epochs          Total epochs to train
#           -b, --batch_size      Dataloader batch size (64)
#           -v, --val_every       Validate every N batches (20)
#           -l, --learning_rate   Learning rate (0.001)
#           -d, --dropout         Dropout (0.2)
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
           python train.py flowers --gpu\n
           python train.py flowers --gpu --arch alexnet --hidden_units 1024 512 -l 0.002 -d 0.1 -e 10\n
           python train.py flowers --gpu --load models/checkpoint_vgg16.pth -e 20"""
    )

    # positional
    
    parser.add_argument('data_dir', type=str,
                        help='Folder of images structured for PyTorch')
    
    # switches
    
    parser.add_argument('--gpu', action='store_true', 
                        help='Use GPU to train the model if CUDA available')

    parser.add_argument('--save_optimizer', action='store_true', 
                        help='Include optimizer state_dict in checkpoint save')

    # options
    
    parser.add_argument('--arch', type=str, default='vgg16', choices=['alexnet','resnet152','vgg16'],
                        help='Pre-trained model to use')
    
    parser.add_argument('--hidden_units', type=int, default=[1024], nargs='+', metavar='int',
                        help='Sizes of hidden layers in classifier ([1024])')
    
    parser.add_argument('-e','--epochs', type=int, default='1', metavar='N',
                        help='Train model for N epochs (1)')
    
    parser.add_argument('-b','--batch_size', type=int, default='64', metavar='N',
                        help='Training data batch size (64)')
    
    parser.add_argument('-v','--val_every', type=int, default='20', metavar='N',
                        help='Validate and output progress every N batches (20)')
    
    parser.add_argument('-l','--learning_rate', type=float,  default='0.001', metavar='N',
                        help='Learning rate (0.001)')
    
    parser.add_argument('-d','--dropout', type=float, default='0.2', metavar='N',
                        help='Dropout on hidden layers in classifier (0.2)')
    
    parser.add_argument('--log', type=str, default='', metavar='file',
                        help='Filename to save screen output')
    
    parser.add_argument('--load', type=str, default='', metavar='file',
                        help='Filename of checkpoint to load and resume training')
    
    parser.add_argument('--save', type=str, default='', metavar='file',
                        help='Filename of checkpoint to save after training')
    
    parser.add_argument('--save_dir', type=str, default='', metavar='folder',
                        help='Folder in which to save checkpoints after training')
    
    parser.add_argument('--save_accuracy', type=float,  default='0.0', metavar='N',
                        help='Start saving best checkpoints after this accuracy is reached')
    
    return parser.parse_args()