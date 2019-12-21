#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Udacity Nanodegree Project
# Image Classifier for AI Programming with Python ND
#
# PROGRAMMER: Michael Gladkowski
# CREATED: 2019-12-15
# REVISED:
# PURPOSE: This class handles sources and pre-processing of 
#          input data for a PyTorch image classifier.  
#          Required by MyClassifier.
#
##

# import modules

import json
from os import path

# external modules

import torch
from torchvision import datasets
from torchvision import transforms
from PIL import Image

# program modules

import output


class MyData:
    '''
        This class handles sources and pre-processing of 
        input data for a PyTorch image classifier.  
        Required by MyClassifier.
        
    '''
    # Properties
    
    dataloaders     = None
    
    data_dir        = ''
    class_to_idx    = {}
    idx_to_class    = {}
    
    names_file      = ''
    cat_to_name     = {}
    
    n_batch_size    = 0
    n_images_train  = 0
    n_images_valid  = 0
    n_images_test   = 0
    n_batches_train = 0
    n_batches_valid = 0
    n_batches_test  = 0
    n_outputs       = 0

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]    


    
    def __init__(self, folder=None, batch_size=None):
        '''
            Constructor
            
            Arguments:
             folder     <str> : Option to load right away
             batch_size <int> : Required if folder
             
        '''
        # option to load training data right away
        
        if folder and batch_size:
            self.load_training_data(folder, batch_size)
            
    
    
    
    def load_training_data(self, folder, batch_size):
        '''
            Loads a folder of training, validation, and test
            images for PyTorch
            
            Arguments:
             folder     <str> : Path to image sets
             batch_size <int> : Training batch size
            
        '''
        # Image folders

        if not folder or not path.exists(folder):
            output.error(f"Invalid data directory : {folder}")
            
        train_dir = path.join(folder,'train')
        valid_dir = path.join(folder,'valid')
        test_dir  = path.join(folder,'test')

        # Define transforms for the training, validation, and testing sets

        data_transforms = {
            'train' : transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(45),
                                          transforms.ToTensor(),
                                          transforms.Normalize(self.mean, self.std)
                                         ]),
            'valid' : transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(self.mean, self.std)
                                         ]),
            'test'  : transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(self.mean, self.std)
                                         ])
        }

        # Load the datasets with ImageFolder

        try:
            image_datasets = {
                'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                'valid' : datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
                'test'  : datasets.ImageFolder(test_dir, transform=data_transforms['test'])
            }
        except Exception as e:
            output.error(f"Torchvision ImageFolder encountered error : {e}")

            
        # Using the image datasets, define the dataloaders

        self.dataloaders = {
            'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
            'valid' : torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size),
            'test'  : torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size)
        }

        # Stop if missing data

        if not self.dataloaders['train']:
            output.error(f"\n\nWarning : No training images found in {train_dir}. Exiting.")

        if not self.dataloaders['valid']:
            output.error(f"\n\nWarning : No validation images found in {valid_dir}. Exiting.")

        if not self.dataloaders['test']:
            output.error(f"\n\nWarning : No test images found in {test_dir}. Exiting.")

        # For convenience some metrics on the above data

        self.data_dir        = folder
        self.n_batch_size    = batch_size
        self.n_outputs       = len(image_datasets['train'].class_to_idx)
        self.n_images_train  = len(image_datasets['train'])
        self.n_images_valid  = len(image_datasets['valid'])
        self.n_images_test   = len(image_datasets['test'])
        self.n_batches_train = len(self.dataloaders['train'])
        self.n_batches_valid = len(self.dataloaders['valid'])
        self.n_batches_test  = len(self.dataloaders['test'])
        self.class_to_idx    = image_datasets['train'].class_to_idx
        self.idx_to_class    = {int(v):int(k) for k,v in self.class_to_idx.items()}

    
    
    
    def load_names(self, filename=''):
        '''
            Load a JSON file with category names into a dictionary
            
            Arguments:
             filename  <string> : Filename of JSON category names
             
        '''
        try:
            with open(filename, 'r') as f:
                self.cat_to_name = json.load(f)
                self.names_file = filename
        
        except Exception as e:
            output.error(f"Error loading category names : {e}")

    
    
    
    def load_image(self, filename):
        ''' 
            Load an image from file using PIL and process
            into PyTorch tensor
            
            Arguments:
             filename <str> : path to an image file
            
            Returns: 
             <tensor> : torch tensor of image ready for feedforward
            
        '''
        # load image
        
        try:
            image = Image.open(filename)
        
        except Exception as e:
            output.error(f"Error loading image : {e}")

        # image transformations
        
        process = transforms.Compose([
                  transforms.Resize(256),
                  transforms.CenterCrop(224),
                  transforms.ToTensor(),
                  transforms.Normalize(self.mean, self.std) 
        ])
        
        # transfom image and return
        
        return process(image).view(1,3,224,224)
    
