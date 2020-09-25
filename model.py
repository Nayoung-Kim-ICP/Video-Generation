import os
import tensorflow as tf
from ops import *
from utils import * 
from BasicConvLSTMCell import BasicConvLSTMCell
import tensorflow.contrib.slim as slim


class MODEL(object):
  def __init__(self, image_size, batch_size, filters,c_dim,num_in, num_out, checkpoint_dir=None, is_train=True):

    self.batch_size = batch_size
    self.image_size = image_size
    self.is_train = is_train
    self.feature_num = 64
    self.df_dim = 32
    self.filters = filters
    self.c_dim = c_dim
    self.num_in = num_in
    self.num_out = num_out
    self.input_shape = [batch_size, num_in,image_size[0],image_size[1],c_dim]
    self.target_shape = [batch_size, num_out, image_size[0],image_size[1],c_dim]
  
    self.build_model()

  def build_model(self):
  
    self.input_ = tf.placeholder(tf.float32, self.input_shape, name='input')
    self.target_ = tf.placeholder(tf.float32, self.target_shape, name='target')     
    filter_unit=[3,3]
    self.rnn_layers = [tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,input_shape=[self.image_size[0]//2, self.image_size[1]//2,self.feature_num*4],kernel_shape=[n, n],output_channels=self.feature_num*4) for n in filter_unit]
    self.multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(self.rnn_layers)
    self.pred_output= self.forward(self.input_)

    self.t_vars = tf.trainable_variables()
    self.saver = tf.train.Saver(max_to_keep=5)  

  def adaptive_conv(self, image, weight):

    color = []
    halffs = int((self.filters-1)//2)
    paddings =tf.constant([[0,0],[halffs,halffs],[halffs,halffs],[0,0]])
    for i in range(self.c_dim):
      input_images=image[:,:,:,i:i+1]
      eximage = tf.extract_image_patches(tf.pad(input_images, paddings, "SYMMETRIC"),ksizes=[1, self.filters, self.filters, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
      output = tf.multiply(eximage, weight)
      output = tf.reduce_sum(output, 3)
      color.append(output)
    return tf.stack(color, 3)

  def networks_enc(self, input_images, reuse):
  
    
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.leaky_relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0001)):
      
      # Define network      
      batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 0.001,
        'is_training': self.is_train,
      }
      with slim.arg_scope([slim.batch_norm], is_training = self.is_train, updates_collections=None):
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params):
         
          self.net1_1 = slim.conv2d(input_images, self.feature_num, [5, 5], stride=1, scope='conv1_1', padding='SAME', reuse = reuse)
          self.net1_2 = slim.conv2d(self.net1_1, self.feature_num, [5, 5], stride=1, scope='conv1_2', padding='SAME', reuse = reuse)
          
          
          self.net2_1 = slim.conv2d(self.net1_2 , self.feature_num*2, [3, 3], stride=1, scope='conv2_1', padding='SAME', reuse = reuse)
          self.net2_2 = slim.conv2d(self.net2_1, self.feature_num*2, [3, 3], stride=1, scope='conv2_2', padding='SAME', reuse = reuse)
          self.net2_p = slim.max_pool2d(self.net2_2, [2, 2], scope='pool2')
        
          self.net3_1 = slim.conv2d(self.net2_p, self.feature_num*4, [3, 3], stride=1, scope='conv3_1', padding='SAME', reuse = reuse)
          self.net3_2 = slim.conv2d(self.net3_1, self.feature_num*4, [3, 3], stride=1, scope='conv3_2', padding='SAME', reuse = reuse)
          self.net3_3 = slim.conv2d(self.net3_2 , self.feature_num*4, [3, 3], stride=1, scope='conv3_3', padding='SAME', reuse = reuse)
          
          
          self.net4_1 = slim.conv2d(self.net3_3, self.feature_num*4, [3, 3], stride=1, scope='conv4_1', padding='SAME', reuse = reuse)
          self.net4_2 = slim.conv2d(self.net4_1, self.feature_num*4, [3, 3], stride=1, scope='conv4_2', padding='SAME', reuse = reuse)
          self.net4_3 = slim.conv2d(self.net4_2, self.feature_num*4, [3, 3], stride=1, scope='conv4_3', padding='SAME', reuse = reuse)
          
         
      self.net_final = slim.conv2d(self.net4_3, self.feature_num*4, [1, 1], stride=1, scope='conv5', activation_fn=tf.tanh, padding='SAME', reuse = reuse)
    return self.net_final




  def networks_dec(self, net,  reuse):

    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.leaky_relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0001)):
      
      # Define network      
      batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 0.001,
        'is_training': self.is_train,
      }
      with slim.arg_scope([slim.batch_norm], is_training = self.is_train, updates_collections=None):
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params):
          
          

          self.net5_1 = slim.conv2d(tf.add(net,self.net_final), self.feature_num*4, [3, 3], stride=1, scope='conv6_1', padding='SAME', reuse = reuse)
          self.net5_2 = slim.conv2d(self.net5_1, self.feature_num*4, [3, 3], stride=1, scope='conv6_2', padding='SAME', reuse = reuse)
          self.net5_3 = slim.conv2d(self.net5_2, self.feature_num*4, [3, 3], stride=1, scope='conv6_3', padding='SAME', reuse = reuse)
         
          
          self.net6_1 = slim.conv2d(self.net5_3, self.feature_num*4, [3, 3], stride=1, scope='conv7_1', padding='SAME', reuse = reuse)
          self.net6_2 = slim.conv2d(self.net6_1, self.feature_num*4, [3, 3], stride=1, scope='conv7_2', padding='SAME', reuse = reuse)
          self.net6_3 = slim.conv2d(self.net6_2, self.feature_num*2, [3, 3], stride=1, scope='conv7_3', padding='SAME', reuse = reuse)

          self.net7_p = tf.image.resize_bilinear(self.net6_3, [self.image_size[0],self.image_size[1]])
          self.net7_1 = slim.conv2d(self.net7_p, self.feature_num*2, [3, 3], stride=1, scope='conv8_1', padding='SAME', reuse = reuse)
          self.net7_2 = slim.conv2d(self.net7_1, self.feature_num, [3, 3], stride=1, scope='conv8_2', padding='SAME', reuse = reuse)

          
          self.net8_1 = slim.conv2d(self.net7_2, self.feature_num*2, [3, 3], stride=1, scope='conv9_1', padding='SAME', reuse = reuse)
          self.net8_2 = slim.conv2d(self.net8_1, self.feature_num, [3, 3], stride=1, scope='conv9_2', padding='SAME', reuse = reuse)
    
    outnet = slim.conv2d(self.net8_2, self.filters*self.filters, [1, 1],stride=1,activation_fn=None, normalizer_fn=None, scope='outout', reuse=reuse)
    output = tf.nn.softmax(outnet)
    
    return output



  def networks_mask(self, first, second, reuse):
    diff = tf.concat([first, second], axis=3)
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.leaky_relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0001)):

      net = tf.image.resize_bilinear(diff, [self.image_size[0]/8,self.image_size[1]/8])
      net = slim.conv2d(net, self.feature_num, [3, 3], stride=1, scope='mask1', padding='SAME', reuse = reuse)
      net = tf.image.resize_bilinear(net, [self.image_size[0]/4,self.image_size[1]/4])
      net = slim.conv2d(net, self.feature_num, [3, 3], stride=1, scope='mask2', padding='SAME', reuse = reuse)
      net = tf.image.resize_bilinear(net, [self.image_size[0]/2,self.image_size[1]/2])
      net = slim.conv2d(net, self.feature_num, [3, 3], stride=1, scope='mask3', padding='SAME', reuse = reuse)
      net = tf.image.resize_bilinear(net, [self.image_size[0],self.image_size[1]])
      net = slim.conv2d(net, self.feature_num, [3, 3], stride=1, scope='mask4', padding='SAME', reuse = reuse)
    mask = slim.conv2d(net, self.c_dim, [1, 1], stride=1,activation_fn = tf.sigmoid, scope='maskr', padding='SAME', reuse = reuse)
    
    return mask


  def networks_lstm(self, net, state):

    with tf.variable_scope("ConvLSTM") as scope:
      
      net , state = self.multi_rnn_cell(net ,state, scope=scope)
    return net, state


  def masking(self, firstframe, nexthat, mask):
    
    net = tf.multiply(mask, nexthat) + tf.multiply(1.0 - mask, firstframe)
    
    return net

  def Network_simple(self,input_):
    #####
    # initialized
    #####  
    pred_out=[]
    reuse =tf.AUTO_REUSE
    first_frame = input_[:,0,:,:,:]
    rescope=False
    self.state = self.multi_rnn_cell.zero_state(self.batch_size, tf.float32)


    first_s = self.networks_enc(first_frame, reuse)

    #####
    # input frame loop
    ##### 
    for n in range(self.num_in):

      if n!=0:
        rescope='True'
  
      input_frame=input_[:,n,:,:,:]
      input_s=self.networks_enc(input_frame, reuse)

      hat_s, self.state = self.networks_lstm(input_s,  self.state)
      
      weight = self.networks_dec(hat_s, reuse)
      hat_frame = self.adaptive_conv(input_frame, weight)
      
      real_hat = hat_frame
      if n==self.num_in-1:
        pred_out.append(real_hat)

    #####
    # output frame loop
    ##### 
    for n in range(self.num_out-1):
      input_s=self.networks_enc(real_hat, reuse)
      hat_s, self.state = self.networks_lstm(input_s,  self.state)      
      weight = self.networks_dec(hat_s,  reuse)

      hat_frame = self.adaptive_conv(real_hat, weight)
      real_hat=hat_frame
      pred_out.append(real_hat)
    return pred_out


  def forward(self,input_):
    pred_out = self.Network_simple(input_)
    return tf.stack(pred_out, axis=1)



  def load(self, sess, checkpoint_dir, model_name=None):
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      if model_name is None: model_name = ckpt_name
      self.saver.restore(sess, os.path.join(checkpoint_dir, model_name))
      print("     Loaded model: "+model_name)
      return True, model_name
    else:
      return False, None

