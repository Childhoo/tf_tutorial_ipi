
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
from tensorflow.python.tools import inspect_checkpoint as chkp

test = tf.constant(3)
ss = tf.Session()
ss.run(test)


# In[2]:

print(ss)


# In[3]:

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

# placeholder for the network input
input_img = tf.placeholder(tf.float32,shape=(None,32,32,3))
labels =  tf.placeholder(tf.float32,shape=(None,10))


# In[4]:

# np.random?
data = np.random.rand(300,32,32,3) #randomly generate data
#generate random labels [one-hot format for np.eye(10)]
values = np.random.randint(10, size=300)
n_values = np.max(values) + 1
print(n_values)
labels_gt = np.eye(n_values)[values]

# start to define the network which is used for classification
#build 3 conv-relu-maxpool layers. The 3rd layer without pooling

# when tensors of variables are created, it is automatically assigned 
# to tf.Graph.Trainable_variables
with tf.variable_scope("1st-layer"):
    output_1 = conv_relu_maxpool(input_img,[5,5,3,10],[10])
with tf.variable_scope("2nd-layer"):
    output_2 = conv_relu_maxpool(output_1,[5,5,10,20],[20])
with tf.variable_scope("3rd-layer"):
    output_3 = conv_relu(output_2,[5,5,20,50],[50])

#build 2 fully connected layers
output_3 = tf.reshape(tf.squeeze(output_3),[-1,50])
fc_1 = tf.layers.dense(output_3, units=200, activation=tf.nn.relu)
fc_2 = tf.layers.dense(fc_1, units=10, activation=tf.nn.relu)


# In[5]:

output_3.shape


# In[8]:
# predictions and losses of the network
predictions = {
    # Generate predictions (for PREDICT and EVAL mode)
    "classes": tf.argmax(input=fc_2, axis=1),
    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
    # `logging_hook`.
    "probabilities": tf.nn.softmax(fc_2, name="softmax_tensor")
}

#define loss of the network
loss = tf.losses.softmax_cross_entropy(labels, fc_2)

#define the training optimizer
train_op = tf.train.AdagradOptimizer(0.001).minimize(loss)
saver = tf.train.Saver()
#saver_part = tf.train.Saver({"fc_1":fc_1,"output_3":output_3})
with tf.Session() as sess:
    writer = tf.summary.FileWriter("/tmp/log/", sess.graph)
    #uncomment this to restore variables from saved checkpoint file 
    saver.restore(sess,"/tmp/log/model.ckpt")
    chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", 
                                          tensor_name='', all_tensors=True)
#    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        feed_dict={input_img:data,labels:labels_gt}
        sess.run(train_op, feed_dict)
        if i%10 ==0:
            print("loss in %5d th step: %.4f" % (i,loss.eval(feed_dict)))
    
    save_path = saver.save(sess, "/tmp/log/model.ckpt")
    print("Model saved in path: %s" % save_path)
    writer.close()



