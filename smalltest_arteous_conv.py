# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 12:09:23 2018

@author: wosleepy
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.tools import inspect_checkpoint as chkp

"""
-------------------------------------------------------------------------------
basic module defined below
"""
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
    variable_summaries(weights)
    variable_summaries(biases)
    variable_summaries(conv)
    tf.summary.histogram('relu_activations', relu)
    return relu

def arteous_conv_relu(input, kernel_shape, bias_shape, input_rate=2):
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    
    conv = tf.nn.atrous_conv2d(input, weights,
        rate=input_rate, padding='SAME')
    relu = tf.nn.relu(conv + biases)
    
    #create some summaries for the variables 
    variable_summaries(weights)
    variable_summaries(biases)
    variable_summaries(conv)
    tf.summary.histogram('relu_activations', relu)
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
    variable_summaries(weights)
    variable_summaries(biases)
    variable_summaries(conv)
    tf.summary.histogram('relu_activations', relu)
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
"""
end of basic module definition
-------------------------------------------------------------------------------
"""
def get_tensor_shape(tensor):

    return [_s if _s is not None else -1 for
            _s in tensor.get_shape().as_list()]

"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
randomly generate training data

"""
# generate random data for test runing of the dataset
data = np.random.rand(300,32,32,3) #randomly generate data
#generate random labels [one-hot format for np.eye(10)]
#values = np.random.randint(10, size=300)

#n_values = np.max(values) + 1
#print(n_values)
#labels_gt = np.eye(n_values)[values]


labels_gt = np.random.rand(300,32,32,1) # random labels for test artous_conv

offset = np.mean(data, 0)
scale = np.std(data, 0).clip(min=1)

data = (data - offset) / scale

"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


"""
-------------------------------------------------------------------------------
define the graph
"""
# start to define the network which is used for classification
#build 3 conv-relu-maxpool layers. The 3rd layer without pooling

# when tensors of variables are created, it is automatically assigned 
# to tf.Graph.Trainable_variables
g1 = tf.Graph()
with g1.as_default():
    # placeholder for the network input
    input_img = tf.placeholder(tf.float32,shape=(None,32,32,3))
    labels =  tf.placeholder(tf.float32,shape=(None,32,32,1))
    

    
    
    tf.summary.image('labels_gt',labels)
    tf.summary.histogram('labels_gt_histogram',labels)
    
    with tf.variable_scope("1st-layer"):
        output_1 = arteous_conv_relu(input_img,[5,5,3,10],[10])
    
    with tf.variable_scope("2nd-layer"):
        output_2 = arteous_conv_relu(output_1,[5,5,10,20],[20])


    with tf.variable_scope("3rd-layer"):
        output_3 = arteous_conv_relu(output_2,[5,5,20,1],[1])
        tf.summary.image('prediction',output_3)
        tf.summary.histogram('prediction_histogram',output_3)
        
    print('shape of predicted result: {}'.format(get_tensor_shape(output_3)))
    

    
#    # predictions and losses of the network
#    predictions = {
#        # Generate predictions (for PREDICT and EVAL mode)
#        "classes": tf.argmax(input=output_3, axis=1),
#        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
#        # `logging_hook`.
#        "probabilities": tf.nn.softmax(fc_2, name="softmax_tensor")
#    }
    
    #define loss of the network
#    loss = tf.losses.softmax_cross_entropy(labels, output_3)
    loss = tf.losses.mean_squared_error(labels, output_3)
    tf.summary.scalar("loss",loss)
    
    #define the training optimizer, it automatically do the forward and backward prop.
    # however, you can also compute the gradient and apply them yourself.
#    train_op = tf.train.AdagradOptimizer(0.001).minimize(loss)
    
    # the second way of processing the gradients (DIY style)
    do_optimize = True
    if do_optimize:
        optimizer = tf.train.AdagradOptimizer(0.001)
        #variables created before were automatically added to trainable_variables...
        grad_var_list = optimizer.compute_gradients(loss, tf.trainable_variables())
        #do some operation to calculated gradients...
        
        new_grads_and_vars = []
        for idx, (grad, var) in enumerate(grad_var_list):
            grad = tf.clip_by_norm(grad, 50) #clip large gradients
            new_grads_and_vars.append((grad, var))
        grad_var_list = new_grads_and_vars
        
        #summary gradients for tensorboard
        for grad, var in grad_var_list:
            tf.summary.histogram(var.name + '/gradient', grad)
        optim_selfbuild = optimizer.apply_gradients(grad_var_list)
        
        #add operations to save and restore all variables in graph
        # passing a tensor list for selected tensor to save is also possible
    saver = tf.train.Saver()
    merge_summary_op = tf.summary.merge_all()
#saver_part = tf.train.Saver({"fc_1":fc_1,"output_3":output_3})
"""
-------------------------------------------------------------------------------
"""    


"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
run graph in a session with the randomly generated data and label
"""
with tf.Session(graph=g1) as sess:
    writer = tf.summary.FileWriter("/tmp/artous_test/log/", sess.graph)

    #if there is pretrained weights, restore them; otherwise train from scratch
    latest_checkpoint = tf.train.latest_checkpoint('/tmp/artous_test/log/')
    if latest_checkpoint is not None:
        saver.restore(sess, latest_checkpoint)
#        chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt",
#                                              tensor_name='', all_tensors=True)
    else:
        sess.run(tf.global_variables_initializer())
            
    for i in range(10000):
        feed_dict={input_img:data, labels:labels_gt}
        if i%5 ==0:
            #summary every 10 steps
            summary, train = sess.run([merge_summary_op, optim_selfbuild], feed_dict)
            writer.add_summary(summary,i)
            writer.flush()
            print("loss in %5d th step: %.4f" % (i,loss.eval(feed_dict)))
        else:
            sess.run(optim_selfbuild, feed_dict)
#    
    # save variables to checkpoint file
    save_path = saver.save(sess, "/tmp/artous_test/log/model.ckpt")
    print("Model saved in path: %s" % save_path)
    writer.close()
"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


