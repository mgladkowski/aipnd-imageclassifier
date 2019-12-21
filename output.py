#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Udacity Nanodegree Project
# Image Classifier for AI Programming with Python ND
#
# PROGRAMMER: Michael Gladkowski
# CREATED: 2019-12-15
# REVISED:
# PURPOSE: Output module for sending program console output
#          to screen and file.
##

logfile = ''


def log(text='', file=''):
    '''
        Write message to desired outputs:
        - screen and filename if passed, or,
        - screen and global logfile, or,
        - screen only if no filename
        
        Arguments:
         text  <str> : The message
         file  <str> : Optional output filepath
        
    '''
    global logfile

    filename = file or logfile
    
    print(text)
    
    if filename:
        try:
            with open(filename, 'a') as f:
                f.write(text+"\n")
                
        except Exception as e:
            print(f"Error writing to logfile : {e}")


            
def error(text='', file=''):
    '''
        Write message to output and terminate program
        
    '''
    log(text, file)
    quit()
