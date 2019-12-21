#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Image Classifier Project for AI Programming with Python Nanodegree
#
# PROGRAMMER: Michael Gladkowski
# CREATED: 2019-12-12
# REVISED:
# PURPOSE: Image Classifier main object
#          Usage for training images:
#         - Load training images using data.load_training_data()
#         - Set GPU if desired with set_device()
#         - Create model with load_checkpoint() or build()
#           with training=True
#         - Set save_to and save_accuracy to save best checkpoint
#         - Now train() and test() the model
#
#          Usage for predicting images:
#         - Load a names file with data.load_names()
#         - Set GPU if desired with set_device()
#         - Create model with load_checkpoint()
#         - predict() any filename
##

# import modules

import time
from collections import OrderedDict
from datetime import datetime
from os import path

# external modules

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from PIL import Image

# program modules

import output
from mydata import MyData
from workspace_utils import active_session



class MyClassifier():
    '''
        Image Classifier Project for AI Programming with Python Nanodegree
        
        Image Classifier main object
        Usage for training images:
       - Load training images using data.load_training_data()
       - Set GPU if desired with set_device()
       - Create model with load_checkpoint() or build()
         with training=True
       - Set save_to and save_accuracy to save best checkpoint
       - Now train() and test() the model

        Usage for predicting images:
       - Load a names file with data.load_names()
       - Set GPU if desired with set_device()
       - Create model with load_checkpoint()
       - predict() any filename
      
    '''
    # Properties
    
    data      = None
    device    = None
    model     = None
    optimizer = None
    criterion = None
    
    # Hyperparameters
    
    arch         = ''
    n_inputs     = 0
    n_outputs    = 0
    hidden_units = []
    epochs       = 0
    epoch        = 0
    step         = 0
    val_every    = 0
    learn_rate   = 0.0
    dropout      = 0.0
    train_stats  = []
    train_loss   = 0.0
    val_loss_min = 0.0
    save_to      = ''
    save_accuracy = 0.0
    save_optimizer = False
    
    # Input sizes of allowed models
    
    input_sizes = { 'alexnet'   : 9216,
                    'resnet152' : 2048,
                    'vgg16'     : 25088 }

    
    
    def __init__(self):
        '''
            Constructor
            
        '''
        # instantiate a data container
        
        self.data = MyData()



        
    def set_device(self, gpu=False):
        '''
            Sets the compute device that PyTorch should use.
            Pass True to use GPU.  If CUDA is unavailable the device 
            will fallback to CPU and output a warning.
            
            Arguments:
             gpu  <bool> : Use CUDA if available
            
        '''
        # assign the device

        self.device = torch.device("cuda" if (gpu and torch.cuda.is_available()) else "cpu")
        str_device = str(self.device)
        
        # warn if a fallback to CPU occurred

        if gpu and str_device == "cpu":
            output.log("CUDA unavailable")

        # output the device in use

        output.log(f"Using {str_device.upper()}")
        output.log()
        



    # Load model
    #
    def load_checkpoint(self, filename, training=False):
        '''
            Restores a model from a checkpoint file, optionally
            including training parameters to resume training
            
            Arguments: 
             filename   <str>   : Checkpoint filename to load
             training   <bool>  : Pass True to restore optimizer state
            
        '''
        try:
            
            # I encountered a few issues with torch.load and devices, this
            # allowed checkpoints to move between part1 and part2 workspaces
            
            if str(self.device) == 'cuda':
                checkpoint = torch.load(filename)
            else:
                checkpoint = torch.load(filename, map_location=str(self.device))
        
        except Exception as e:
            output.error(f"Error loading checkpoint : {e}")

        # get what we need to build the architecture

        self.arch         = checkpoint['arch']
        self.n_inputs     = checkpoint['n_inputs']
        self.n_outputs    = checkpoint['n_outputs']
        self.hidden_units = checkpoint['hidden_units']
        self.epochs       = checkpoint['epochs']
        self.epoch        = checkpoint['epoch']
        self.step         = checkpoint['step']
        self.learn_rate   = checkpoint['learn_rate']
        self.dropout      = checkpoint['dropout']
        self.train_loss   = checkpoint['train_loss']
        self.val_loss_min = checkpoint['val_loss']
        self.data.class_to_idx = checkpoint['class_to_idx']
        self.data.idx_to_class = {int(v):int(k) for k,v in checkpoint['class_to_idx'].items()}
        
        # create the model
        
        self.build(self.arch, self.hidden_units, self.learn_rate, self.dropout, training=training)
        
        # now we can load the state_dict for model

        try:
            self.model.load_state_dict(checkpoint['state_dict'])
        
        except Exception as e:
            output.error(f"Error loading model.state_dict : {e}\n{self.model}")
            
        # load state_dict for optimizer if applicable
        
        if training and 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
            except Exception as e:
                output.error(f"Error loading optimizer.state_dict : {e}\n{self.optimizer}")
                
        # output confirmation
        
        if checkpoint and self.model:
            output.log(f"Loaded model checkpoint {filename} size {path.getsize(filename)}")
            output.log()
            
    
    
    
    def save_checkpoint(self, filename):
        '''
            Saves a model checkpoint with parameters needed to resume training
            
            Arguments:
             filename <str> : Path of checkpoint file to save
             include_optimizer <bool> : Pass True to save optimizer.state_dict
             
        '''
        checkpoint = {
            'arch'         : self.arch,
            'n_inputs'     : self.n_inputs,
            'n_outputs'    : self.n_outputs,
            'epochs'       : self.epochs,
            'epoch'        : self.epoch,
            'step'         : self.step,
            'learn_rate'   : self.learn_rate,
            'dropout'      : self.dropout,
            'train_loss'   : self.train_loss,
            'val_loss'     : self.val_loss_min,
            'hidden_units' : self.hidden_units,
            'class_to_idx' : self.data.class_to_idx,
            'state_dict'   : self.model.state_dict()
        }

        # optimizer gradients optional due to size
        
        if self.save_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()

        # save the checkpoint
        
        try:
            torch.save(checkpoint, filename)
            
        except Exception as e:
            output.error(f"Error saving checkpoint : {e}")
            
    
    
    
    def build(self, arch, hidden_units, learning_rate=0.0, dropout=0.0, training=True):
        '''
            Builds a new classifier from a pre-trained network and custom classifier
            
            Arguments:
             arch         <str>   : pre-trained architecture (alexnet|reset152|vgg16)
             hidden_units <list>  : Python list of integer hidden layer sizes
             learn_rate   <float> : Learn rate
             dropout      <float> : Dropout
             training     <bool>  : Pass True to add optimizer
                
        '''
        # validate arguments
        
        if not arch in self.input_sizes:
            output.error(f"Error building model : {arch} is unsupported, choose alexnet|resnet152|vgg16")

        # assign hyperparameters
        
        self.arch         = arch
        self.n_inputs     = self.input_sizes[arch]
        self.n_outputs    = self.n_outputs or self.data.n_outputs
        self.hidden_units = hidden_units
        self.learn_rate   = learning_rate
        self.dropout      = dropout

        # load pre-trained model by name
        
        network = getattr(models, arch)(pretrained=True)
        
        # peel off existing classifier, need its name
        
        pname,pclass = network._modules.popitem()
        
        # freeze pre-trained parameters
        
        for p in network.parameters():
            p.requires_grad = False
        
        # classifier input layer

        layers = []
        layers.append( ('input', nn.Linear(self.n_inputs, hidden_units[0])) )
        layers.append( ('relu_0', nn.ReLU()) )
        layers.append( ('drop_0', nn.Dropout(self.dropout)) )
        
        # hidden layers
        
        for i in range(1, len(hidden_units)):
            layers.append( (f"layer{i}", nn.Linear(hidden_units[i-1], hidden_units[i])) )
            layers.append( (f"relu_{i}", nn.ReLU()) )
            layers.append( (f"drop_{i}", nn.Dropout(self.dropout)) )
        
        # output layer
        
        layers.append( ('layerz', nn.Linear(hidden_units[-1], self.n_outputs)) )
        layers.append( ('output', nn.LogSoftmax(dim=1)) )
        
        
        # create classifier
        
        network.add_module(pname, nn.Sequential(OrderedDict(layers)))

        if training:
            
            # gradient descent optimizer
            self.optimizer = optim.Adam(getattr(network, pname).parameters(), lr=self.learn_rate)

            # error loss function
            self.criterion = nn.NLLLoss()

        
        # network ready
        
        self.model = network

        # move to compute device
        
        self.model.to(self.device)

    
    
    
    def validation(self, dataset):
        '''
            Run validation
            
            Arguments:
             dataset <str> : A dataloader key, 'valid' or 'test'
            
            Returns:
             val_loss <float>
             accuracy <float>
             
        '''
        val_loss = 0.0
        accuracy = 0.0

        # run model on dataset

        for images,labels in self.data.dataloaders[dataset]:

            # move data to compute device
            images,labels = images.to(self.device), labels.to(self.device)

            # feed forward
            logps = self.model(images)
            loss = self.criterion(logps,labels)
            val_loss += loss.item()

            # turn logarithmic probabilities back into probabilities
            ps = torch.exp(logps)

            # get top predictions
            top_ps,top_class = ps.topk(1, dim=1)

            # determine accuracy vs label
            equality = top_class == labels.view(*top_class.shape)

            # compute accuracy
            accuracy += float(torch.mean(equality.type(torch.FloatTensor)))
        
        return val_loss,accuracy

    
    
    
    def train(self, epochs, val_every):
        '''
            Run training
            
            Arguments:
             epochs    <int> : Number of epochs to train
             val_every <int> : Perform validation every N batches
            
        '''
        # assign hyperparameters
        
        self.epochs     = epochs
        self.val_every  = val_every
        
        # display templates
        
        layout_vh = "{:^8} {:<6} {:>12} {:>12} {:>12} {:>10} {:>8}"
        layout_vr = "{:^3d}/{:<4d} {:<6d} {:>12.3f} {:>12.3f} {:>12.3f} {:>10} {:>8}"

        # keep Udacity session active

        with active_session():

            # begin training

            start_date = datetime.now().strftime("%d-%m-%Y %H:%M:%S UTC")
            start_time = time.time()
            
            # display start information
            
            output.log(f"Started training  : {start_date}")
            output.log(f"Model             : {self.arch}")
            output.log()
            output.log(f"Training images   : {self.data.n_images_train}")
            output.log(f"Validation images : {self.data.n_images_valid}")
            output.log(f"Testing images    : {self.data.n_images_test}")
            output.log(f"Outputs           : {self.data.n_outputs}")
            output.log(f"Hidden layers     : {self.hidden_units}")
            output.log(f"Epochs            : {self.epochs}")
            output.log(f"Validation every  : {self.val_every} batches")
            output.log(f"Learn rate        : {self.learn_rate}")
            output.log(f"Dropout           : {self.dropout}")
            output.log()
            
            if self.epoch > 0:
                output.log(f"Resuming training epoch {self.epoch} of {self.epochs}")
                output.log()
                
            if self.save_to:
                output.log(f"Checkpoints will be saved to {self.save_to}")
                output.log()
                
            output.log(layout_vh.format('Epoch','Step','Train loss','Val loss','Accuracy','Elapsed','Saved'))
            
            # training mode

            self.model.train()

            # iterate through epochs
            
            while self.epoch < self.epochs:

                # epoch counter
                self.epoch += 1

                # iterate batches
                for images,labels in self.data.dataloaders['train']:

                    # batch counter
                    self.step += 1
                    
                    # move data to device
                    images,labels = images.to(self.device), labels.to(self.device)

                    # zero the gradients
                    self.optimizer.zero_grad()

                    # feed forward
                    logps = self.model(images)

                    # compute error
                    loss = self.criterion(logps, labels)

                    # backpropagate
                    loss.backward()

                    # gradient descent takes a step
                    self.optimizer.step()

                    # track total error
                    self.train_loss += loss.item()


                    # validate and output progress every N steps

                    if self.step % self.val_every == 0:

                        val_loss = 0.0
                        accuracy = 0.0
                        
                        # evaluation mode, dropout and gradients off
                        
                        self.model.eval()
                        with torch.no_grad():

                            # validation
                            val_loss,accuracy = self.validation('valid')
                                
                        # current stats
                        self.train_loss /= self.val_every
                        val_loss /= self.data.n_batches_valid
                        accuracy /= self.data.n_batches_valid

                        # first pass initializes min validation loss
                        self.val_loss_min = self.val_loss_min or val_loss

                        # save minimum validation losses above accuracy threshold
                        saved=''
                        if val_loss < self.val_loss_min and accuracy >= self.save_accuracy:
                            self.val_loss_min = val_loss
                            self.save_checkpoint(self.save_to)
                            saved='Y'

                            
                        # update and output training stats
                        elapsed_time = time.time() - start_time
                        elapsed_str  = time.strftime('%H:%M:%S',time.gmtime(elapsed_time))
                        
                        self.train_stats.append([self.epoch, self.step, self.train_loss, 
                                                 val_loss, accuracy, elapsed_str ])

                        output.log(layout_vr.format(self.epoch, self.epochs, self.step, self.train_loss,
                                                    val_loss, accuracy, elapsed_str, saved ))

                        # reset training loss
                        self.train_loss = 0.0

                        # back to training mode
                        self.model.train()

        # finished training
        finish_date = datetime.now().strftime("%d-%m-%Y %H:%M:%S UTC")

        output.log(f"\nTraining complete : {finish_date}")
        output.log()


    
    
    def test(self):
        '''
            Run testing
            
        '''
        test_loss = 0
        accuracy  = 0

        # display templates
        
        layout_th = "{:15} {:>12} {:>12}"
        layout_tr = "{:15} {:>12.3f} {:>12.3f}"
        
        output.log(layout_th.format('Testing','Test loss','Accuracy'))
        
        # evaluation mode, dropout and gradients off
        
        self.model.eval()
        with torch.no_grad():

            # validation on test set
            
            test_loss,accuracy = self.validation('test')

        # output test results
        
        output.log(layout_tr.format(self.arch, 
                                    test_loss/self.data.n_batches_test, 
                                    accuracy/self.data.n_batches_test))
        output.log()
        
        
        
        
    def forward(self, image_tensor, topk=5):
        ''' 
            Get predictions for an image tensor
            
            Arguments:
             image_tensor  <tensor> : An image tensor
             topk          <int>    : Output top K predictions
            
            Returns:
             probs   <list> : List of confidence probabilities
             classes <list> : List of predicted classes
            
        '''
        # move to compute device
        image_tensor = image_tensor.to(self.device)

        # evaluation mode, dropout and gradients off
        
        self.model.eval()
        with torch.no_grad():
            
            # feed forward
            output = self.model(image_tensor)

            # output probabilities
            ps = torch.exp(output)

            # top k probabilities
            top_ps,top_class = ps.topk(topk, dim=1)

        # return outputs as python lists
        probs = top_ps.cpu().detach().tolist()[0]
        classes = top_class.cpu().detach().tolist()[0]
        
        return probs,classes

    
    
    
    def predict(self, filename, topk=5):
        '''
            Run prediction on an input image and
            output the results
            
            Arguments:
             filename  <string> : Path to input image
             topk      <int>    : Output top K probabilities
                
        '''
        # get image tensor
        
        image_tensor = self.data.load_image(filename)
        
        # run prediction
        
        probs,classes = self.forward(image_tensor, topk)
        
        # display results
        
        output.log(f"Input file    : {filename}")
        output.log("{:>5} {:<25} {:>10}".format('Class','Name','Confidence'))

        # iterate over top_k results and output
        
        for i in range(len(classes)):
        
            # retrieve the class and its name if available
            
            i_index = classes[i]
            i_class = self.data.idx_to_class[i_index]
            i_name  = self.data.cat_to_name.get(str(i_class), '').title()
            
            output.log(f"{i_class:>5d} {i_name:<25} {probs[i]*100:10.4f} %")

        output.log()
