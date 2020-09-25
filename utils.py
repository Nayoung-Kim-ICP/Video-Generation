"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import cv2
import random
import imageio
import scipy.misc
import numpy as np
import os
import pdb



def transform(image):
    return image/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

def save_images(images, size, image_path):
  return imsave(inverse_transform(images)*255., size, image_path)


def merge(images, size):
  
  img = np.zeros((size[0], size[1]*size[2], 3))
  #x = size[2]
  for idx in range(size[2]):
    img[:,idx*size[1]:(idx+1)*size[1],:]=images[:,:,idx,:]
    
  return img

def merge_img(images, size):
  
  h = images.shape[0]
  w = images.shape[1]
  img = np.zeros((h,w*size,3))
  for k in range(images.shape[2]):
    img[:,k*w:k*w+w,:]=images[:,:,k,:]
    
  return img


def imsave(images, size, path):
  return scipy.misc.imsave(path, merge(images, size))


"""
def imsave(np_image, filename,size):
  #Save image to file.
  #Args:
  #  np_image: .
  #  filename: .
  #
  # im = sp.misc.toimage(np_image, cmin=0, cmax=1.0)
  im = sp.misc.toimage(merge(np_image,size), cmin=-1.0, cmax=1.0)
  im.save(filename)
"""
def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])



def crop_center(img,cropx,cropy):

    y = img.shape[1]
    x = img.shape[2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[:,starty:starty+cropy,startx:startx+cropx,:]

def crop_front(img,cropy,cropx):

    y = img.shape[1]
    x = img.shape[2]
    starty = y-cropy
    startx = x-cropx
    return img[:,starty:,startx:,:]




def get_minibatches_idx(n, minibatch_size, shuffle=False):
  """ 
  Used to shuffle the dataset at each iteration.
  """

  idx_list = np.arange(n, dtype="int32")

  if shuffle:
    random.shuffle(idx_list)
    random.shuffle(idx_list)
  minibatches = []
  minibatch_start = 0 
  for i in range(n // minibatch_size):
    minibatches.append(idx_list[minibatch_start:
                                minibatch_start + minibatch_size])
    minibatch_start += minibatch_size

  if (minibatch_start != n): 
    # Make a minibatch out of what is left
    minibatches.append(idx_list[minibatch_start:])

  return zip(range(len(minibatches)), minibatches)


def draw_frame(img, is_input):
  if img.shape[2] == 1:
    img = np.repeat(img, [3], axis=2)

  if is_input:
    img[:2,:,0]  = img[:2,:,2] = 0 
    img[:,:2,0]  = img[:,:2,2] = 0 
    img[-2:,:,0] = img[-2:,:,2] = 0 
    img[:,-2:,0] = img[:,-2:,2] = 0 
    img[:2,:,1]  = 255 
    img[:,:2,1]  = 255 
    img[-2:,:,1] = 255 
    img[:,-2:,1] = 255 
  else:
    img[:2,:,0]  = img[:2,:,1] = 0 
    img[:,:2,0]  = img[:,:2,2] = 0 
    img[-2:,:,0] = img[-2:,:,1] = 0 
    img[:,-2:,0] = img[:,-2:,1] = 0 
    img[:2,:,2]  = 255 
    img[:,:2,2]  = 255 
    img[-2:,:,2] = 255 
    img[:,-2:,2] = 255 

  return img 

