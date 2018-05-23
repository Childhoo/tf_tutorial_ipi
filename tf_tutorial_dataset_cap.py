# -*- coding: utf-8 -*-
"""
Created on Tue May 22 17:03:25 2018

@author: chen.admin
"""
import numpy as np 
from tf_tutorial_dataset_cifar import load_dataset, classify_name

class IPI_CIFAR_Dataset:
    def __init__(self):
        self._index_in_epoch_train = 0
        self._epochs_completed_train = 0
        self._index_in_epoch_valid = 0
        self._epochs_completed_valid = 0
        
#        self._data = load_dataset()
        
        # load dataset for each module
        self.train_data, self.train_label, \
        self.valid_data, self.valid_label, \
        self.test_data,  self.test_label =  load_dataset()
        
        # get the number of train/val samples
        self._num_examples_train = self.train_data.shape[0]
        self._num_examples_valid = self.valid_data.shape[0]
        pass
    
    
    @property
    def get_train_data(self):
        return self.train_data
    
    def next_batch(self,batch_size,shuffle = True, mode = 'train'):
        if mode == 'train':
            start = self._index_in_epoch_train
            if start == 0 and self._epochs_completed_train == 0:
                idx = np.arange(0, self._num_examples_train)  # get all possible indexes
                np.random.shuffle(idx)  # shuffle indexe
                self._train_data = self.train_data[idx]  # get list of `num` random samples
                self._train_label = self.train_label[idx]
                
            # go to the next batch
            if start + batch_size > self._num_examples_train:
                self._epochs_completed_train += 1
                rest_num_examples_train = self._num_examples_train - start
                data_rest_part_train = self._train_data[start:self._num_examples_train]
                label_rest_part_train = self._train_label[start:self._num_examples_train]
                idx0 = np.arange(0, self._num_examples_train)  # get all possible indexes
                np.random.shuffle(idx0)  # shuffle indexes
                self._train_data = self.train_data[idx0]  # get list of `num` random samples
                self._train_label = self.train_label[idx0]
                
                start = 0
                self._index_in_epoch_train = batch_size - rest_num_examples_train #avoid the case where the #sample != integar times of batch_size
                end =  self._index_in_epoch_train  
                data_new_part_train =  self._train_data[start:end]  
                label_new_part_train =  self._train_label[start:end]  
                return np.concatenate((data_rest_part_train, data_new_part_train), axis=0),\
                        np.concatenate((label_rest_part_train, label_new_part_train), axis=0) \
            
                                      
            else:
                self._index_in_epoch_train += batch_size
                end = self._index_in_epoch_train
                return self._train_data[start:end], self._train_label[start:end]
        
        if mode == 'valid':
            start = self._index_in_epoch_valid
            if start == 0 and self._epochs_completed_valid == 0:
                idx = np.arange(0, self._num_examples_valid)  # get all possible indexes
                np.random.shuffle(idx)  # shuffle indexe
                self._valid_data = self.valid_data[idx]  # get list of `num` random samples
                self._valid_label = self.valid_label[idx]
        
            # go to the next batch
            if start + batch_size > self._num_examples_valid:
                self._epochs_completed_valid += 1
                rest_num_examples = self._num_examples_valid - start
                data_rest_part = self._valid_data[start:self._num_examples_valid]
                label_rest_part = self._valid_label[start:self._num_examples_valid]
                idx0 = np.arange(0, self._num_examples_valid)  # get all possible indexes
                np.random.shuffle(idx0)  # shuffle indexes
                self._valid_data = self.valid_data[idx0]  # get list of `num` random samples
                self._valid_label = self.valid_label[idx0]
        
                start = 0
                self._index_in_epoch_valid = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
                end =  self._index_in_epoch_valid  
                data_new_part =  self._valid_data[start:end]  
                label_new_part = self._valid_label[start:end]  
                return np.concatenate((data_rest_part, data_new_part), axis=0),\
                        np.concatenate((label_rest_part, label_new_part), axis=0)
            else:
                self._index_in_epoch_valid += batch_size
                end = self._index_in_epoch_valid
                return self._valid_data[start:end], self._valid_label[start:end]



#dataset = IPI_CIFAR_Dataset()
#for i in range(10):
#    print(dataset.next_batch(128))