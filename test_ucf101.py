import cv2
import sys
import time
import imageio
import pdb

import tensorflow as tf
import scipy.misc as sm
import numpy as np
import scipy.io as sio
import io
from os import *
from model  import *
from utils import *
from dataloader import *
from ops import *
from argparse import ArgumentParser
from os.path import exists


def main(lr, batch_size,image_size,iters,epochs,filter_size,c_dim,num_in,num_out,interpolation,gpu, model_name):
  is_train=False
  os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu[0])
  hfilters = (filter_size-1)//2
  data_path = "./UCF-101(FRAME)/"
  f = io.open(data_path+"pushup.txt","r")
  trainfiles = f.readlines()
  
  prefix  = ("train1"+"_filter_size_"+str(filter_size))
  
  print("\n"+prefix+"\n")
  checkpoint_dir = "./checkpoint/"+prefix+"/"
  results_dir = "./result/"+prefix+"/"

  if not exists(results_dir ):
    makedirs(results_dir )

  image_size=[64,64]
  with tf.device("/gpu:%d"%gpu[0]):
    pdb.set_trace()  
    model = MODEL(is_train=is_train,image_size=image_size, c_dim= c_dim, batch_size=batch_size, num_in=num_in, num_out=num_out,checkpoint_dir=checkpoint_dir, filters = filter_size)

  
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)) as sess:
    tf.global_variables_initializer().run()
    
    if model.load(sess, checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

  
    counter = 1
    start_time = time.time()
    epoch_num = epochs
    psnr_final = 0
    ssim_final = 0
    
    start_time = time.time() 
    iternum=iters
    for epoch in range(epoch_num):
      iternum=0
      mini_batches = get_minibatches_idx(len(trainfiles), 1, shuffle=False)
  
      for _, batchidx in mini_batches:
        tfiles = np.array(trainfiles)[batchidx]
        for m in range(1):
          batch_input, batch_output = load_ucf101_10_class_test(tfiles, data_path,image_size, num_in, num_out,batch_size,m)   
          feed_dict_ =  {model.input_:batch_input,model.target_ : batch_output }
           
          prediction_np = sess.run([model.pred_output],feed_dict = feed_dict_)[0]
          samplepred = np.transpose(inverse_transform(prediction_np[0,:,:,:,:]),(1,2,0,3))
          sampletarg = np.transpose(inverse_transform(batch_output[0,:,:,:,:]),(1,2,0,3))
          
          pred = merge_img(samplepred,num_out)
          target = merge_img(sampletarg, num_out)
          psnr_total = psnr(pred,target)
          ssim_total = ssim(pred, target)
          print("[video %d] PSNR : %4.4f , SSIM: %4.4f"%(counter, psnr_total, ssim_total))
          save_name = results_dir+"%s_tr.png"%(counter)
          cv2.imwrite(save_name, target*255)
          counter = counter+1
          
if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--lr", type=float, dest="lr",
                      default=0.00001, help="Base Learning Rate")
  parser.add_argument("--batch_size", type=int, dest="batch_size",
                      default=1, help="Mini-batch size")
  parser.add_argument("--c_dim", type=int, dest="c_dim",
                      default=3, help="image channel")
  parser.add_argument("--iters", type=int, dest="iters",
                      default=1, help="iters num")
  parser.add_argument("--epochs", type=int, dest="epochs",
                      default=1, help="epochs num")
  parser.add_argument("--image_size", type=int, dest="image_size",
                      default=64, help="Inpput_image_size")
  parser.add_argument("--filter_size", type=int, dest="filter_size",
                      default=15, help="filtersize")
  parser.add_argument("--num_in", type=int, dest="num_in",
                      default=10, help="Number of steps to observe from the past")
  parser.add_argument("--num_out", type=int, dest="num_out",
                      default=10, help="Number of steps into the future")
  parser.add_argument("--gpu", type=int, nargs="+", dest="gpu", required=True,
                      help="GPU device id")
  parser.add_argument("--type", type=bool, dest="interpolation",
                      default=False, help="interpolation:True, extrapolation:False")
  parser.add_argument("--model_name", type=str, dest="model_name",
                      default="spatio", help="write name")

  args = parser.parse_args()
  main(**vars(args))
