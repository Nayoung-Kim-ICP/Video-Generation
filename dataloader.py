import cv2
import random
import imageio
import scipy.misc
import scipy as sp
import numpy as np
import os


def transform(image):
  return image/127.5 - 1.



def get_minibatches_idx(n, minibatch_size, shuffle=True):
  """ 
  Used to shuffle the dataset at each iteration.
  """

  idx_list = np.arange(n, dtype="int32")

  if shuffle:
    random.shuffle(idx_list)
    random.shuffle(idx_list)
    random.shuffle(idx_list)

  else:
    np.sort(idx_list)
    np.sort(idx_list)
    np.sort(idx_list)

  minibatches = []
  minibatch_start = 0 
  for i in range(n // minibatch_size):
    minibatches.append(idx_list[minibatch_start:
                                minibatch_start + minibatch_size])
    minibatch_start += minibatch_size

  if (minibatch_start != n): 
    minibatches.append(idx_list[minibatch_start:])

  return zip(range(len(minibatches)), minibatches)



def make_frame(w_h, w_w,inputs,h,w,c_in, filters):
  output_frame=np.zeros((h,w,c_in),dtype=np.float64)
  for i in range(h):
    for j in range(w):
      hw=np.reshape(w_h[i,j,:],(filters,1))
      ww=np.reshape(w_h[i,j,:],(1,filters))
      weight = np.matmul(hw,ww)
      block= inputs[i:filters+i,j:filters+j,:]
      for c in range(c_in):
        color = np.multiply(block[:,:,c], weight)
        output_frame[i,j,c]=np.sum(color)
  return output_frame      




def load_data_test_frame(f_name, data_path,image_size, num_in, num_out,batch_size,point, filters):
  input_frame = np.zeros((batch_size,num_in,image_size[0],image_size[1],3),dtype = np.float64)
  output_frame = np.zeros((batch_size,num_out,image_size[0],image_size[1],3),dtype = np.float64)
  
  for i in range(batch_size):
    
    vid_path = os.path.join(data_path ,f_name)
    vid_path = vid_path[:-1]
    
    name = "/"+str(point)+".png"
    image = cv2.imread(vid_path+name)
    image = image.astype("float32")

    
    for hh in range(num_in):
      name = "/"+str(point+hh)+".png"
      image = cv2.imread(vid_path+name)
      image = image.astype("float32")
      input_frame[i,hh,:,:,:]= image

      
    for hh in range(num_in,num_out+num_in):
      name = "/"+str(point+hh)+".png"
      image = cv2.imread(vid_path+name)
      image = image.astype("float32")
      output_frame[i,hh-num_in,:,:,:]= image

  return transform(input_frame),transform(output_frame)




def load_ucf101_10_class_test(f_name, data_path,image_size, num_in,num_out,batch_size,m):
  input_frame = np.zeros((batch_size,num_in,image_size[0],image_size[1],3),dtype = np.float64)
  output_frame = np.zeros((batch_size,num_out,image_size[0],image_size[1],3),dtype = np.float64)

  for i in range(batch_size):
    vid_name = f_name[i]
    vid_path = os.path.join(data_path ,vid_name)
    vid_path = vid_path[:-1]
    point = (m)+1


    for hh in range(num_in):
      name1 ="/"+str(point+hh)+".png"     
      image1 = cv2.imread(vid_path+name1)
      image1 = cv2.resize(image1, (image_size[0],image_size[1]), interpolation = cv2.INTER_CUBIC)
      image1 = image1.astype("float32")

      input_frame[i,hh,:,:,:]=image1

      for hh in range(num_in, num_out+num_in):    
        name1 = "/"+str(point+hh)+".png"     
        image1 = cv2.imread(vid_path+name1)
        image1 = cv2.resize(image1, (image_size[0], image_size[1]), interpolation = cv2.INTER_CUBIC )
        image1 = image1.astype("float32")
        output_frame[i,hh-num_in,:,:,:]=image1
  return transform(input_frame), transform(output_frame)

