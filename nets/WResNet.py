from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import Distillation as Dist

tcf = tf.contrib.framework
tcl = tf.contrib.layers

def BasicBlock(x, out_planes, stride = 2, keep_rate = 1., name = None):
    with tf.variable_scope(name):
        equalInOut = x.get_shape().as_list()[-1] == out_planes
        if not equalInOut:
            x_ = tcl.batch_norm(x, scope='bn')
            x = tf.nn.relu(x_)
            if name[-1] == '0':
                tf.add_to_collection('feat', x)
        else:
            out_ = tcl.batch_norm(x, scope='bn')
            out = tf.nn.relu(out_)
            if name[-1] == '0':
                tf.add_to_collection('feat', out)
                
        out = tcl.batch_norm(tcl.conv2d(out if equalInOut else x, out_planes, [3,3], stride, scope='conv0'), activation_fn = tf.nn.relu, scope='bn0')
        out = tcl.dropout(out, keep_rate)

        out = tcl.conv2d(out, out_planes, [3,3], 1, scope='conv1')
        if not(equalInOut):
            x = tcl.conv2d(x, out_planes, [1,1], stride, scope='conv2')
        return x+out
    
def NetworkBlock(x, nb_layers, out_planes, stride, keep_rate, name = ''):
    with tf.variable_scope(name):
        for i in range(nb_layers):           
            x = BasicBlock(x, out_planes, stride = stride if i == 0 else 1, keep_rate = keep_rate, name = 'BasicBlock%d'%i)
        return x

def WResNet(image, label, scope, is_training, Distill = None):
    end_points = {}
    is_training, auxiliary_is_training = is_training

    if scope == 'Teacher':     
        depth = 40; widen_factor = 4
    else:
        depth = 16; widen_factor = 4
    
    nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
    stride = [1,2,2]
    n = (depth-4)//6
    
    with tf.variable_scope(scope):
        with tcf.arg_scope([tcl.conv2d, tcl.fully_connected, tcl.batch_norm], trainable = True):
            with tcf.arg_scope([tcl.dropout, tcl.batch_norm], is_training = is_training):
                if image.get_shape().as_list()[1] == 32:
                    std = tcl.conv2d(image, nChannels[0], [3,3], 1, scope='conv0')
                    keep_rate = .7
                else:
                    std = tcl.conv2d(image, nChannels[0], [7,7], 2, scope='conv0')
                    keep_rate = 1.
                
                for i in range(3): 
                    std = NetworkBlock(std, n, nChannels[1+i], stride[i], keep_rate, name = 'WResblock%d'%i)
                std = tcl.batch_norm(std, scope='bn0')
                std = tf.nn.relu(std)
                tf.add_to_collection('feat', std)
                fc = tf.reduce_mean(std, [1,2])
                
                logits = tcl.fully_connected(fc , label.get_shape().as_list()[-1],
                                             biases_initializer = tf.zeros_initializer(),
                                             scope = 'full')
                end_points['Logits'] = logits
        
    if Distill is not None:
        teacher_trainable = False
        teacher_is_training = False
            
        arg_scope = WResNet_arg_scope_teacher()
            
        with tf.variable_scope('Teacher'):
            with tcf.arg_scope(arg_scope):
                with tcf.arg_scope([tcl.conv2d, tcl.fully_connected, tcl.batch_norm], trainable = teacher_trainable):
                    with tcf.arg_scope([tcl.dropout, tcl.batch_norm], is_training = teacher_is_training):
                        depth = 40; widen_factor = 4 # teacher
                        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
                        n = (depth-4)//6
                        stride = [1,2,2]
                        
                        if image.get_shape().as_list()[1] == 32:
                            tch = tcl.conv2d(image, nChannels[0], [3,3], 1, scope='conv0')
                        else:
                            tch = tcl.conv2d(image, nChannels[0], [7,7], 2, scope='conv0')
                        keep_rate = 1.
                    
                        for i in range(3):            
                            tch = NetworkBlock(tch, n, nChannels[1+i], stride[i], keep_rate, name = 'WResblock%d'%i)
                        tch = tcl.batch_norm(tch,scope='bn0')
                        tch = tf.nn.relu(tch)
                        tf.add_to_collection('feat', tch)
    
                        fc = tf.reduce_mean(tch, [1,2])
                        logits_tch = tcl.fully_connected(fc , label.get_shape().as_list()[-1], 
                                                         biases_initializer = tf.zeros_initializer(),
                                                         scope = 'full')
                        end_points['Logits_tch'] = logits_tch
                    
        with tf.variable_scope('Distillation'):
            feats = tf.get_collection('feat')
            student_feats = feats[1:len(feats)//2]
            teacher_feats = feats[1+len(feats)//2:]
            Dist.EKI(student_feats, teacher_feats)

    return end_points

def WResNet_arg_scope():
    with tcf.arg_scope([tcl.conv2d, tcl.fully_connected], 
                       weights_initializer=tcl.variance_scaling_initializer(mode='FAN_OUT'),
                       biases_initializer=None, activation_fn = None):
        with tcf.arg_scope([tcl.batch_norm], scale = True, center = True, activation_fn=None, decay=0.9, epsilon = 1e-5,
                           variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES,'BN_collection']) as arg_sc:
            return arg_sc
            
def WResNet_arg_scope_teacher():
    with tcf.arg_scope([tcl.conv2d, tcl.fully_connected], 
                       variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES,'Teacher']):
        with tcf.arg_scope([tcl.batch_norm], 
                           variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES,'Teacher']) as arg_sc:
            return arg_sc
            

