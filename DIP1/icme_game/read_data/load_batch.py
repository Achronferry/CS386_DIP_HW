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

def data_preprocess(input):        
    res = []
    t_l = []
    i = input[0]
    j = input[1]
    l = input[2]
    r1 = crop(i)
    r2 = crop(j)
    if l == '1':
      l = 1
    else:
      l = 0
    for k in range(len(r1)):
        t_l.append(l)
        res.append(np.concatenate((r1[k],r2[k]),axis = 1))
    return np.array(res),np.array(t_l)


def Tobatch(res,batch_size):
    data,label = res[0],res[1]
    d1 = []
    la = []
    length = data.shape[0]
    index = 0
    while index+batch_size<length:
        d1.append(data[index:index+batch_size])
        la.append(label[index:index+batch_size])
        index+=batch_size
    d1.append(data[index:index+batch_size])
    la.append(label[index:index+batch_size])
    return d1,la

def get_batch(choose=0, batch_size=32):
    dataset = "/home/PublicStore/lanyuting/project/CS386_DIP_HW/DIP1/icme_game/"+str(choose)+".txt"
    myfile = open(dataset,'r')
    lines = myfile.readlines()
    index = random.randint(0,len(lines)-1)
    #print(index)
    return Tobatch(data_preprocess(lines[index][:-1].split(" ")),batch_size)
 
def Data_filtering(train_index, beta=5, m=0):
  dataset_dir='/home/PublicStore/lanyuting/project/dataset/'
  myfile1 = open(str(m)+".txt","w")
  mode = ['Training/','Test/']
  for index in range(1,train_index):
    batch = []  
    image_dir = dataset_dir + mode[0]+ str(index).rjust(3,'0')+"/"  
    index_score = {}
    #score_dir = dataset_dir+ "score_and_sort/Training/score/"+str(index).rjust(3,'0')+"_score.csv"
    score_dir = dataset_dir+ "score_and_sort/Training/sort/"+str(index).rjust(3,'0')+".csv"
    print(image_dir)
    print(score_dir)
    #input()
    myfile = open(score_dir,'r')
    for row in myfile:
      mes = row[:-1].split(',')
      if mes[1] == "Color":
        continue
      index_score[mes[0]] = [float(mes[1]),float(mes[2]),float(mes[3]),float(mes[4])]
    print(index_score)
    #input()
    list_dirs = os.walk(image_dir)
    label=[]
    for root, dirs, files in list_dirs: 
        for file in files:
          batch.append(image_dir+file)
          label.append(index_score[file])
    
    for index1 in range(14):
      for index2 in range(14):
        if index1!=index2:
          if label[index1][m] > label[index2][m] + beta:
            l = "0"
            print(batch[index1]+' '+batch[index2]+' '+l)
            print(label[index1][m])
            print(label[index2][m])
            myfile1.write(batch[index1]+' '+batch[index2]+' '+l+'\n')
          else:
            if label[index1][m] < label[index2][m] - beta:
              l = "1"
              print(batch[index1]+' '+batch[index2]+' '+l)
              print(label[index1][m])
              print(label[index2][m])
              myfile1.write(batch[index1]+' '+batch[index2]+' '+l+'\n')

    # print(batch)
    # print(label)

if __name__ == "__main__":
  Data_filtering(70,7,0)
  Data_filtering(70,7,1)
  Data_filtering(70,7,2)
  Data_filtering(70,7,3)
  print(0,get_batch()[0][0].shape)
  print(1,get_batch()[1][0].shape)