# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:13:58 2018

@author: Lin Chen
"""

import tensorflow as tf
# example comes from official tutorial 
# https://www.tensorflow.org/programmers_guide/tensors

# define a placeholder 
p = tf.placeholder(shape = (None), dtype = tf.float32)
# run some operations with placeholders
t = p + 1.0
mul = tf.multiply(p,p)


# write the default graph 
writer = tf.summary.FileWriter('./logs')
writer.add_graph(tf.get_default_graph())
writer.close()


#t.eval()  # This will fail, since the placeholder did not get a value
res = t.eval(session = tf.Session(), feed_dict={p:2.0})  
# This will succeed because we're feeding a value
                           # to the placeholder.

print('result from tensor eval:')
print(res)                  

sess = tf.Session()
print('result from session run:')
print(sess.run(t, feed_dict={p:2.0}))
# tensor:eval; operations:run; sess.run(..)

print('result of (p+1, p*p):')
print(sess.run((t,mul), feed_dict={p:2.0}))

sess.close() #close session when used without with block