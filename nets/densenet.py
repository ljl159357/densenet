"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current


def block(net, layers, growth, scope='block'):
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net


def densenet(images, num_classes=1001, is_training=False,
             dropout_keep_prob=0.8,
             scope='densenet'):
    """Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    growth = 24
    compression_rate = 0.5

    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:
            ##########################
            # Put your code here.
            
            "Initial_convolution"
            end_point = 'Initial_convolution'
            with tf.variable_scope(end_point):
		            x = slim.conv2d(images, growth, [7,7], stride=2, padding='same', scope='conv7x7')
		            x = slim.batch_norm(x, scope='bn')
								x = tf.nn.relu(x)	
								x = slim.max_pool2d(x, [3, 3], stride=2,  padding='same', scope='MaxPool3x3')
								net = slim.dropout(x, scope='dropout')
            end_points[end_point]= net
            
           "dense_block1"
            end_point = 'dense_block1'
            with tf.variable_scope(end_point):
		            net=block(net, 6, growth, scope=end_point)    
            end_points[end_point]= net            
            
            "transition_lay1"
            end_point = 'transition_lay1'
            with tf.variable_scope(end_point):
		            x=bn_act_conv_drp(net, reduce_dim(net), [1,1], scope='conv1x1')
								net = slim.avg_pool2d(x, [2,2] stride=2,  scope='avgPool2x2')
            end_points[end_point]= net            
            
            
           "dense_block2"
            end_point = 'dense_block2'
            with tf.variable_scope(end_point):
		            net=block(net, 12, growth, scope=end_point)    
            end_points[end_point]= net            
            
            "transition_lay2"
            end_point = 'transition_lay2'
            with tf.variable_scope(end_point):
		            x=bn_act_conv_drp(net, reduce_dim(net), [1,1], scope='conv1x1')
								net = slim.avg_pool2d(x, [2,2] stride=2,  scope='avgPool2x2')
            end_points[end_point]= net                
            
           "dense_block3"
            end_point = 'dense_block3'
            with tf.variable_scope(end_point):
		            net=block(net, 24, growth, scope=end_point)    
            end_points[end_point]= net            
            
            "transition_lay3"
            end_point = 'transition_lay3'
            with tf.variable_scope(end_point):
		            x = bn_act_conv_drp(net, reduce_dim(net), [1,1], scope='conv1x1')
								net = slim.avg_pool2d(x, [2,2] stride=2,  scope='avgPool2x2')
            end_points[end_point]= net              
            
           "dense_block4"
            end_point = 'dense_block4'
            with tf.variable_scope(end_point):
		            net = block(net, 16, growth, scope=end_point)    
            end_points[end_point]= net              
            
           "global_pool"
            end_point = 'global_pool'  
            with tf.variable_scope(end_point): 
            	net = slim.batch_norm(net,scope='bn')
            	net = tf.nn.relu(net)        
            	net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
            	net = slim.dropout(net, scope='Dropout')
        		end_points[end_point] = net
            
           "Logits"
            end_point = 'Logits'  
            with tf.variable_scope(end_point):            
            		logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')   
            		logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')    
                end_points['predictions'] = slim.softmax(logits, scope='predictions')
          	end_points[end_point] = logits
            
            
            ##########################

    return logits, end_points


def bn_drp_scope(is_training=True, 
								keep_prob=0.8,
								batch_norm_decay=0.997,
								batch_norm_epsilon=1e-5,
								batch_norm_scale=True
								):
									
    keep_prob = keep_prob if is_training else 1   
		batch_norm_params = {
				'decay': batch_norm_decay,
				'epsilon': batch_norm_epsilon,
				'scale': batch_norm_scale,
				'updates_collections': tf.GraphKeys.UPDATE_OPS,
				'fused': None,  # Use fused batch norm if possible.
		}
    with slim.arg_scope([slim.batch_norm], is_training=is_training, **batch_norm_params):
        with slim.arg_scope([slim.dropout], is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False), 
        biases_initializer=tf.zeros_initializer(), 
        weights_regularizer=slim.l2_regularizer(weight_decay) ):
        		with slim.arg_scope([slim.conv2d], activation_fn = None, normalizer_fn = None, padding='same',stride=1) as sc:
  							return sc


densenet.default_image_size = 224
