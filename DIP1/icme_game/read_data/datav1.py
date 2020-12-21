from os import listdir
from os.path import join
from PIL import Image
import torch.utils.data as data
import os
import numpy as np
import torch
import pandas as pd
from random import randrange
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def is_image_file(filename):
  filename_lower = filename.lower()
  return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.bmp', '.mat'])

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir='/home/whr/dataset', transform=None):
        super(DatasetFromFolder, self).__init__()
        self.data_dir = "{}/training_set/".format(image_dir)
        self.dir_filenames = [join(self.data_dir, x) for x in listdir(self.data_dir)]
        self.dir_file_num = len(self.dir_filenames)
        self.transform = transform
        self.crop_width = 1024
        self.crop_height = 1024

    def __getitem__(self, index):
        dir_path_name = self.dir_filenames[index]
        path_list = os.listdir(dir_path_name)
        path_list.sort()
        
        path_csv = join(dir_path_name, path_list[0])
        pd_data = pd.read_csv(path_csv).Texture
        label = np.array(1-(pd_data/15))
        # label = np.array(pd_data)
        label = torch.from_numpy(label)
        
        path_list = [join(dir_path_name, x) for x in path_list if is_image_file(x)]

        data_init = Image.open(path_list[0])
        width, height = data_init.size
        x, y = randrange(0, width - self.crop_width + 1), randrange(0, height - self.crop_height + 1)
        data_init = data_init.crop((x, y, x + self.crop_width, y + self.crop_height))
        #data_init = data_init.resize((4032, 3024))
        if self.transform:
            data_init = self.transform(data_init)

        for i in range(len(path_list)-1):
            data = Image.open(path_list[i+1])
            width, height = data.size
            x1, y1 = randrange(0, width - self.crop_width + 1), randrange(0, height - self.crop_height + 1)
            data = data.crop((x1, y1, x1 + self.crop_width, y1 + self.crop_height))
            #data = data.resize((4032, 3024))
            if self.transform:
                data = self.transform(data)
            data_init = torch.cat((data_init, data), 0)

        return data_init, label



    def __len__(self):
        return len(self.dir_filenames)


class TestFromFolder(data.Dataset):
    def __init__(self, image_dir='/home/whr/dataset', transform=None):
        super(TestFromFolder, self).__init__()
        self.data_dir = "{}/testing_set/".format(image_dir)
        self.dir_filenames = [join(self.data_dir, x) for x in listdir(self.data_dir)]
        self.dir_file_num = len(self.dir_filenames)
        self.transform = transform


    def __getitem__(self, index):
        dir_path_name = self.dir_filenames[index]
        path_list = os.listdir(dir_path_name)
        path_list.sort()

        name = path_list[0].split('.')[0]

        path_csv = join(dir_path_name, path_list[0])
        pd_data = pd.read_csv(path_csv).Exposure
        label = np.array(pd_data)
        label = torch.from_numpy(label).float()

        path_list = [join(dir_path_name, x) for x in path_list if is_image_file(x)]
        data_init = Image.open(path_list[0])
        data_init = data_init.resize((1024, 1024))

        if self.transform:
            data_init = self.transform(data_init)
            name =int(name)

        for i in range(len(path_list) - 1):
            data = Image.open(path_list[i + 1])
            data = data.resize((1024, 1024))
            if self.transform:
                data = self.transform(data)
            data_init = torch.cat((data_init, data), 0)

        return data_init, label, name

    def __len__(self):
        return len(self.dir_filenames)


