from os import listdir
from os.path import join
from PIL import Image
import torch.utils.data as data
import os
import numpy as np
import torch
import pandas as pd
from random import randrange
from PIL import ImageFile,Image
import random
import csv
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import io, transform

def is_image_file(filename):
  filename_lower = filename.lower()
  return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.bmp', '.mat'])

def crop(path_csv, resize = 224):
    
    result = []
    #load and resize image
    im = Image.open(path_csv)
    width, height = im.size
    im1 = im.resize((3136,4032))
    
    data = np.array(im1)

    x1 = 0
    x2 = resize
    y1 = 0
    y2 = resize
    while y2<=data.shape[1]:
      if x2 <= data.shape[0]:
        result.append(data[x1:x2,y1:y2])
        x1+=resize
        x2+=resize
      else:
        x1 = 0
        x2 = resize
        y1 += resize
        y2 += resize
    return result

def data_preprocess(image_list,label,index1,index2):        
    res = []
    t_l = []
    i = image_list[index1]
    j = image_list[index2]
    r1 = crop(i)
    r2 = crop(j)
    for k in range(len(r1)):
        l = []
        for m in range(4):
          if label[index1][m] > label[index2][m]:
            l.append(0)
          else:
            l.append(1)
        t_l.append(l)
        res.append(np.concatenate((r1[k],r2[k]),axis = 1))
    return np.array(res),np.array(t_l)

def get_batch(train_index = 70, valid_index = 30):
    dataset_dir='/home/PublicStore/lanyuting/project/dataset/'
    mode = ['Training/','Test/']
    batch = []
    index = random.randint(1,train_index)
    image_dir = dataset_dir + mode[0]+ str(index).rjust(3,'0')+"/"  

    index_score = {}
    score_dir = dataset_dir+ "score_and_sort/Training/score/"+str(index).rjust(3,'0')+"_score.csv"
    myfile = open(score_dir,'r')
    for row in myfile:
      mes = row[:-1].split(',')
      if mes[0] == "Name":
        continue
      index_score[mes[0]] = [float(mes[1]),float(mes[2]),float(mes[3]),float(mes[4])]
    list_dirs = os.walk(image_dir)
    label=[]
    for root, dirs, files in list_dirs: 
        for file in files:
          batch.append(image_dir+file)
          label.append(index_score[file])
    while True:
      index1 = random.randint(0,14)
      index2 = random.randint(0,14)
      if index1!=index2:
        break
    return data_preprocess(batch,label,index1,index2)
 


if __name__ == "__main__":
  get_batch()