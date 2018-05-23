# -*- coding: utf-8 -*-
"""
Created on Tue May 22 23:42:59 2018

@author: Lin Chen
"""
import tensorflow as tf

def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='VALID')
    relu = tf.nn.relu(conv + biases)
    
    #create some summaries for the variables 
#    variable_summaries(weights)
#    variable_summaries(biases)
#    variable_summaries(conv)
#    tf.summary.histogram('relu_activations', relu)
    return relu

def conv_relu_maxpool(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='VALID')
    relu = tf.nn.relu(conv + biases)
    maxpool = tf.nn.max_pool(relu,[1,2,2,1],
                             strides=[1, 2, 2, 1],
                             name='max-pool',
                             padding='VALID')
#    variable_summaries(weights)
#    variable_summaries(biases)
#    variable_summaries(conv)
#    tf.summary.histogram('relu_activations', relu)
    return maxpool

#define a function for summary variables
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

