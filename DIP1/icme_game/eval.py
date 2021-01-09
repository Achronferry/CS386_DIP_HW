import torch
import numpy as np
import sys
sys.path.append("..")
import scipy
import torch.optim as optim
import os
from PIL import Image
# a = [1,2,3,4,5,6]
# b = [1,2,3,4,5,6]

# print(scipy.stats.spearmanr(a, b)[0])

# optimizer = 
# path1 = "/home/PublicStore/lanyuting/project/dataset/Training/001/L_001.jpg"
# path2 = "/home/PublicStore/lanyuting/project/dataset/Training/001/E_001.jpg" 

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


def Tobatch(res,batch_size):
    data = res
    d1 = []
    length = data.shape[0]
    index = 0
    while index+batch_size<length:
        d1.append(data[index:index+batch_size])
        index+=batch_size
    d1.append(data[index:index+batch_size])

    return d1

def sort(model,path,device,batch_size = 16):
    index = 1
    test_dir = path + str(index).rjust(3,'0')
    list_dirs = os.walk(test_dir)
    for root,dirs,files in list_dirs:
        length = len(files)
        print(files)
  
        for i in range(1,length):
            for j in range(length-i):
                res=[]
                path1 = os.path.join(root,files[j])
                path2 = os.path.join(root,files[j+1])
                r1 = crop(path1)
                r2 = crop(path2)
                print(len(r1))
                for k in range(len(r1)):
                    res.append(np.concatenate((r1[k],r2[k]),axis = 1))
 
                res = np.array(res)
                res = Tobatch(res,batch_size)
                sum1 = np.array([0.0,0.0])
                for batch_data in res:
                    batch_data = torch.from_numpy(batch_data.transpose(0,3,1,2)).float().to(device)
                    outputs = model(batch_data)
                    sum1 += torch.sum(outputs,axis=0).detach().cpu().numpy()
                print(sum1)
                if sum1[0] > sum1[1]:
                    files[j+1],files[j] = files[j],files[j+1]
    print(files)
          
                
        
   



if __name__ == "__main__":
    # evaluate()
    model = torch.load("model/resnet50.pt")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    path = "/home/PublicStore/lanyuting/project/dataset/Training/"
    #path = "/home/PublicStore/lanyuting/project/dataset/Test/"
    sort(model,path,device)