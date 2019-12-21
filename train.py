#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Image Classifier Project for AI Programming with Python Nanodegree
#
# PROGRAMMER: Michael Gladkowski
# CREATED: 2019-12-12
# REVISED:
# PURPOSE: Trains a network on any set of images of any set of classes.
#          The architecture consists of a choice of pre-trained torchvision
#          network, with a custom classifier that we train.  An images folder
#          is supplied along with a class-to-human dictionary.
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
#
#   Examples:
#   New default network:
#   python train.py flowers --gpu
#
#   New custom network:
#   python train.py flowers --gpu --arch alexnet -e 10 -b 32 -v 20 -h 2048 512 -l 0.002 -d 0.1
#
#   Resume training a checkpoint:
#   python train.py flowers --gpu --load models/checkpoint_vgg16.pth -e 20 --log results.txt --save_dir models --save_optimizer
#
##

# import modules

from os import path
from datetime import datetime

# program modules

import output
import train_args
from myclassifier import MyClassifier



def main():
    '''
        Main program
    '''
    # get command line arguments
    
    args = train_args.get_input_args()

    # output header
    
    output.logfile = args.log
    output.log("Image Classifier Project for AI Programming with Python Nanodegree")
    output.log()

    # new model with training data
    
    classifier = MyClassifier()
    
    classifier.data.load_training_data(args.data_dir, args.batch_size)
    
    # set compute device
    
    classifier.set_device(args.gpu)

    # load checkpoint to resume if provided
    
    if args.load and path.exists(args.load):
        classifier.load_checkpoint(args.load, training=True)
        
    # otherwise build a new model from parameters
    
    else:
        classifier.build(args.arch, args.hidden_units, args.learning_rate, args.dropout, training=True)
        
    # apply checkpoint save arguments
    
    classifier.save_to = args.save or path.join(args.save_dir,f"checkpoint_{classifier.arch}.pth")
    classifier.save_accuracy = args.save_accuracy
    classifier.save_optimizer = args.save_optimizer
    
    # train the model
    
    classifier.train(args.epochs, args.val_every)

    # test the model
    
    classifier.test()



# RUN PROGRAM
#
if __name__ == "__main__":
    main()
    