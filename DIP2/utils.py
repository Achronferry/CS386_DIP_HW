import os, sys
import numpy as np
import random
import cv2
import torch



def save_saliencymap(saliencymap, image_filename, output_path):
    EXTENSION = 'pgm'

    image_name = ('.').join(os.path.basename(image_filename).split('.')[:-1])
    filename = "%s/%s.%s" % (output_path, image_name, EXTENSION)
    cv2.imwrite(filename, saliencymap)




def calc_loss(pred, truth, bias, alpha=1.1, beta=0.1):
    '''
        pred: S'-[bsize, w, h] prediction result; 
        truth: S-[bsize, w, h] ground truth;
        bias: B-[bsize, w, h] represents the positional bias observed with ASD people; simply the
                            average saliency maps computed over the training dataset;

        Loss = avg( (s-s')^2/(alpha-s) + beta*(s'-b)^2 )
    '''
    s_, s, b = pred.view(pred.shape[0],-1), truth.view(pred.shape[0],-1), bias.view(pred.shape[0],-1)
    l1 = (s-s_)**2 / (alpha * torch.ones_like(s) - s)
    l2 = beta * (s_-b)**2
    loss = torch.mean(l1+l2, dim=-1,keepdim=False)
    return loss



class Dataset():
    def __init__(self, images, sailiency_maps, shape):
        self.shape = shape
        self.images = np.stack([self.preprocess_images(i, self.shape) for i in images])

        self.origin_sizes = [i.shape for i in sailiency_maps]
        self.maps = np.stack([cv2.resize(i, self.shape[::-1]).astype(np.float32) for i in sailiency_maps])
        self.len = len(self.images)
  

    def preprocess_images(self, original_image, shape):
        image = cv2.resize(original_image, shape[::-1]).astype(np.float32)
        # Remove train image mean (imagenet)
        image[:, :, 0] -= 103.939
        image[:, :, 1] -= 116.779
        image[:, :, 2] -= 123.68

        image = image.transpose(2,0,1)
        return image


    def batch(self, batch_size, device=torch.device('cpu')):
        ids = list(range(self.len))
        random.shuffle(ids)

        ptr = 0
        while ptr+batch_size < self.len:
            batch_x = torch.tensor(self.images[ids[ptr:ptr+batch_size]]).to(device)
            batch_y = torch.tensor(self.maps[ids[ptr:ptr+batch_size]]).to(device)
            yield batch_x, batch_y
            ptr += batch_size
        


    def __len__(self):
        return self.len

def load_dataset(path, shape=(300,400), test_rate=0.2):
    image_path = os.path.join(path,'Images')
    fixmap_path = os.path.join(path, 'ASD_FixMaps')
    image_names = list(filter(lambda x: x[-4:]!='png', os.listdir(image_path)))
    image_names.sort(key=lambda k: int(k[:-4]))

    images, sailiency_maps = [], []
    for i in image_names:
        i_path = os.path.join(image_path, i)
        s_path = os.path.join(fixmap_path, i.replace('.png','_s.png'))
        images.append(cv2.imread(i_path))
        sailiency_maps.append(cv2.imread(s_path, cv2.COLOR_BGR2GRAY))
        
    split_num = int(len(images) * (1-test_rate))
    train_set = Dataset(images[:split_num], sailiency_maps[:split_num], shape)
    test_set = Dataset(images[split_num:], sailiency_maps[split_num:], shape)

    return train_set, test_set


if __name__=='__main__':
    a,b = load_dataset('dataset/TrainingData')
    print(len(a))
    print(len(b))
    for x,y in a.batch(10):
        print(x.shape)
        print(y.shape)
    pass


