import tensorflow as tf
import numpy as np
from tensorflow.python.tools import inspect_checkpoint as chkp
from tf_tutorial_dataset_cap import IPI_CIFAR_Dataset
from tf_tutorial_basic_module import conv_relu, conv_relu_maxpool


# instead of use random generated data, here we will use real data (CIFAR)
# necessary methods are wrapped in the class IPI_CIFAR_Dataset.
dataset = IPI_CIFAR_Dataset()

# start to define the network which is used for classification
#build 3 conv-relu-maxpool layers. The 3rd layer without pooling

"""
-------------------------------------------------------------------------------
define the graph
"""
g1 = tf.Graph()
with g1.as_default():
    # placeholder for the network input
    input_img = tf.placeholder(tf.float32,shape=(None,32,32,3))
    labels =  tf.placeholder(tf.float32,shape=(None,10))
    
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
    tf.summary.scalar("loss",loss)
    #define the training optimizer, it automatically do the forward and backward prop.
    # however, you can also compute the gradient and apply them yourself.
#    train_op = tf.train.AdagradOptimizer(0.001).minimize(loss)
    
    # the second way of processing the gradients
    
    optimizer = tf.train.AdagradOptimizer(0.0001)
    #variables created before were automatically added to trainable_variables...
    grad_var_list = optimizer.compute_gradients(loss, tf.trainable_variables())
    #do some operation to calculated gradients...
    
    #clip large gradients
    new_grads_and_vars = []
    for idx, (grad, var) in enumerate(grad_var_list):
        grad = tf.clip_by_norm(grad, 50) 
        new_grads_and_vars.append((grad, var))
    grad_var_list = new_grads_and_vars
    
    #summary gradients for tensorboard
    for grad, var in grad_var_list:
        tf.summary.histogram(var.name + '/gradient', grad)
        tf.summary.histogram(var.op.name, var)
    optim_selfbuild = optimizer.apply_gradients(grad_var_list)
    
    #add operations to save and restore all variables in graph
    # passing a tensor list for selected tensor to save is also possible
    saver = tf.train.Saver()
    merge_summary_op = tf.summary.merge_all()
    
    decoded_one_hot_label = tf.argmax(labels, axis=1)
    correct = tf.nn.in_top_k(fc_2, decoded_one_hot_label, 1)
    eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
"""
-------------------------------------------------------------------------------
"""


    
"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
run graph in a session with CIFAR Data
"""
with tf.Session(graph=g1) as sess:
    writer = tf.summary.FileWriter("/tmp/log/", sess.graph)

    #if there is pretrained weights, restore them; otherwise train from scratch
    latest_checkpoint = tf.train.latest_checkpoint('/tmp/log/')
    if latest_checkpoint is not None:
        saver.restore(sess, latest_checkpoint)
        print('loaded weights from pretrained model')
    else:
        print('train from scratch')
        sess.run(tf.global_variables_initializer())
    
    best_validation_prec = 0.0          
    for i in range(10000):
        data_batch, lable_batch = dataset.next_batch(128, mode='train')
        feed_dict={input_img:data_batch, labels:lable_batch}
        
        if i%10 == 0:
            #summary every 10 steps
            summary, train = sess.run([merge_summary_op, optim_selfbuild], feed_dict)
            writer.add_summary(summary,i)
            writer.flush()
            print("loss in %5d th step: %.4f" % (i,loss.eval(feed_dict)))
        # do validation every 50 steps
        if i%200 == 0: 
            valid_batch, valid_label = dataset.next_batch(2000, mode='valid')
            feed_dict_val={input_img:valid_batch, labels:valid_label}
            fetch_valid = {"loss":loss,
                           "eval_correct":eval_correct}
            res_valid = sess.run(fetch_valid, feed_dict_val)
            
            # add validation loss and precision into summary
            summary_valid_loss = [tf.Summary.Value(
                    tag="validation/loss",
                    simple_value=res_valid["loss"],
            )]
            summary_valid_prec = [tf.Summary.Value(
                    tag="validation/precision",
                    simple_value=res_valid["eval_correct"]/2000,
            )]
            writer.add_summary(tf.Summary(value=summary_valid_loss),i)
            writer.add_summary(tf.Summary(value=summary_valid_prec),i)
            writer.flush()
            print("valid precision in %5d th step: %.4f" 
                  % (i, res_valid["eval_correct"]/2000))
            
            # check whether it is better than the best validated trained model
            if res_valid["eval_correct"]/2000 > best_validation_prec:
                # save best validated training result
                save_path = saver.save(sess, "/tmp/log/model.ckpt")
                print("best validated model saved in path: %s" % save_path)
                best_validation_prec = res_valid["eval_correct"]/2000
        else:
            sess.run(optim_selfbuild, feed_dict)

"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""