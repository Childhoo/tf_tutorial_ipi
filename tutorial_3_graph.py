

# -*- coding: utf-8 -*-
"""
Created on Tue May 22 11:14:23 2018

@author: Lin Chen
"""



import tensorflow as tf
# example comes from official tutorial 
# https://www.tensorflow.org/programmers_guide/tensors

# define a placeholder and the graph
g1 = tf.Graph()
with g1.as_default():
    p = tf.placeholder(shape = (None), dtype = tf.float32, name='input')
    #t = p + 1.0
    t = tf.add(p, 1.0, name='plus_1') 
    # operations also take "tensor-like" object such as scalar
    # graph is more readable when change + to tf operations
    mul = tf.multiply(p, p, name='square')
    # giving names will make the graph more readable.
    '''
    this part writes the default graph into an events files
    '''
    # write the default graph 
    writer = tf.summary.FileWriter('./logs')
    writer.add_graph(tf.get_default_graph())
    writer.close()




#t.eval()  # This will fail, since the placeholder did not get a value
#session = tf.Session(graph=g1)
res = t.eval(feed_dict={p:2.0}, session=tf.Session(graph=g1))
print('result from tensor eval:')
print(res)   

               
# execute in some sessions
with tf.Session(graph=g1) as sess:
    
    print('result from session run:')
    print(sess.run(t, feed_dict={p:2.0}))
    # tensor:eval; operations:run; sess.run(..)
    
    print('result of [p+1, p*p]:')
    print(sess.run([t,mul], feed_dict={p:2.0}))


#sess.close() #no need to close sess anymore when used in with block

# print operations in default graph
#for op in g1.get_operations():
#    print(op)