import torch
import numpy as np
import sys
sys.path.append("..")
from read_data import load_batch
import scipy
import torch.optim as optim
a = [1,2,3,4,5,6]
b = [1,2,3,4,5,6]

print(scipy.stats.spearmanr(a, b)[0])
model = torch.load("model/resnet1.pt")


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)
# optimizer = 
# path1 = "/home/PublicStore/lanyuting/project/dataset/Training/001/L_001.jpg"
# path2 = "/home/PublicStore/lanyuting/project/dataset/Training/001/E_001.jpg" 


def evaluate(figure1=None,figure2=None,choose=0):
    #return bool
    #choose = 0,1,2,3, which means for type
    valid_data,valid_labels = load_batch.get_batch(choose, batch_size=32)
    for i in range(len(valid_data)):
        batch_data = valid_data[i]  
        batch_label = valid_labels[i]
        batch_data = torch.from_numpy(batch_data.transpose(0,3,1,2)).float().to(device)
        batch_label = torch.from_numpy(batch_label).long().to(device)

        outputs = model(batch_data)
        print(outputs.size)
        input()
        # print(batch_label.size())
        # print(outputs.size())

        size += batch_data.size(0)



if __name__ == "__main__":
    evaluate()