import numpy as np
import cv2
import math
import torch


def calculate_auc_j(gt_, s_map_):
    """
    Calculate AUC-Judd.
    """
    # print(gt_, s_map_, np.max(gt_))
    gt_ = torch.from_numpy(gt_).unsqueeze(dim=0)
    s_map_ = torch.from_numpy(s_map_).unsqueeze(dim=0)
    batch_size = gt_.size(0)

    ret_ = 0.

    gt_ = torch.where(gt_ > 0, torch.ones_like(gt_), torch.zeros_like(gt_))

    for i_ in range(batch_size):

        #  ===========================
        # print(torch.max(s_map_), torch.max(gt_))

        gt = gt_[i_].detach().cpu().numpy()
        s_map = s_map_[i_].detach().cpu().numpy()
        # ground truth is discrete, s_map is continous and normalized
        # thresholds are calculated from the salience map, only at places where fixations are present
        thresholds = []
        for i in range(0, gt.shape[0]):
            for k in range(0, gt.shape[1]):
                if gt[i][k] > 0:
                    thresholds.append(s_map[i][k])

        num_fixations = np.sum(gt)
        # num fixations is no. of salience map values at gt >0

        thresholds = sorted(set(thresholds))

        # fp_list = []
        # tp_list = []
        area = []
        area.append((0.0, 0.0))
        for thresh in thresholds:
            # in the salience map, keep only those pixels with values above threshold
            temp = np.zeros(s_map.shape)
            temp[s_map >= thresh] = 1.0
            assert np.max(gt) <= 1.0, 'something is wrong with ground truth..not discretized properly max value > 1.0'
            assert np.max(s_map) <= 1.0, 'something is wrong with salience map..not normalized properly max value > 1.0'
            num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
            tp = num_overlap / (num_fixations * 1.0)

            # total number of pixels > threshold - number of pixels that overlap with gt / total number of non fixated pixels
            # this becomes nan when gt is full of fixations..this won't happen
            fp = (np.sum(temp) - num_overlap) / ((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)

            area.append((round(tp, 4), round(fp, 4)))
        # tp_list.append(tp)
        # fp_list.append(fp)

        # tp_list.reverse()
        # fp_list.reverse()
        area.append((1.0, 1.0))
        # tp_list.append(1.0)
        # fp_list.append(1.0)
        # print tp_list
        area.sort(key=lambda x: x[0])
        tp_list = [x[0] for x in area]
        fp_list = [x[1] for x in area]
        # print(np.trapz(np.array(tp_list), np.array(fp_list)))
        ret_ += np.trapz(np.array(tp_list), np.array(fp_list))
    return ret_ / batch_size

def AUC_Borji(saliencyMap, fixationMap,Nsplits=100, stepSize=0.1, toPlot=0):
    if np.sum(fixationMap)<=1:
        print('no fixationMap')
        return 
    score=0
    saliencyMap=saliencyMap.astype(float)
    fixationMap=fixationMap.astype(float)
    if saliencyMap.shape != fixationMap.shape:
        saliencyMap = cv2.resize(saliencyMap,(fixationMap.shape[1],fixationMap.shape[0]))
    saliencyMap = (saliencyMap-np.min(saliencyMap))/(np.max(saliencyMap)-np.min(saliencyMap))
    S=saliencyMap
    S=np.reshape(S,S.shape[0]*S.shape[1],order='F')
    F=fixationMap
    F=np.reshape(F,F.shape[0]*F.shape[1],order='F')	
    Sth=S[np.where(F>0)]
    Nfixations=len(Sth)
    Npixels=len(S)
    r=np.random.randint(Npixels,size=(Nfixations,Nsplits))
    randfix=S[r]
    
    auc=[0]*Nsplits
    for s in range(Nsplits):
        curfix=randfix[:,s]
        temp=list(Sth)
        temp.extend(list(curfix))
        allthreshes=x2 = np.arange(0,np.max(temp)+stepSize,stepSize)
        allthreshes=allthreshes[::-1]
        tp=np.zeros(len(allthreshes)+2)
        fp=np.zeros(len(allthreshes)+2)
        tp[0]=0
        tp[-1]=1
        fp[0]=0
        fp[-1]=1
        for i in range(len(allthreshes)):
            thresh=allthreshes[i]
            tp[i+1]=np.sum(len(np.where(Sth>= thresh)[0]))/(float)(Nfixations)
            fp[i+1]=np.sum(len(np.where(curfix>= thresh)[0]))/(float)(Nfixations)
        auc[s]=np.trapz(x=fp,y=tp)
        #print(auc)
    score=np.mean(auc)
    return score

def AUC_Judd(saliencyMap, fixationMap,jitter=0, toPlot=0):
    score=0
    saliencyMap=saliencyMap.astype(float)
    fixationMap=fixationMap.astype(float)
    if saliencyMap.shape != fixationMap.shape:
        saliencyMap = cv2.resize(saliencyMap,(fixationMap.shape[1],fixationMap.shape[0]))
    if jitter:
        saliencyMap = saliencyMap+np.random.rand(fixationMap.shape[0],fixationMap.shape[1])/10000000.0
    saliencyMap=(saliencyMap-np.min(saliencyMap))/(np.max(saliencyMap)-np.min(saliencyMap))
    S=saliencyMap        
    S=np.reshape(S,S.shape[0]*S.shape[1],order='F')
    F=fixationMap        
    F=np.reshape(F,F.shape[0]*F.shape[1],order='F')
    Sth=S[np.where(F>0)]
    Nfixations = len(Sth)
    Npixels = len(S)        
    allthreshes = np.sort(Sth, axis=None)
    allthreshes = allthreshes[::-1]
    tp = np.zeros(Nfixations+2)
    fp = np.zeros(Nfixations+2)
    tp[0] = 0
    tp[-1] = 1
    fp[0] = 0
    fp[-1] = 1
    for i in range(0,Nfixations):
        thresh = allthreshes[i]
        aboveth = np.sum(len(np.where(S>= thresh)[0]))
        tp[i+1] = (float)(i) / Nfixations
        fp[i+1] = (float)(aboveth-i) / (Npixels - Nfixations)

    score = np.trapz(x=fp,y=tp)
    return score


def CC(saliencyMap1, saliencyMap2):

    def mean2(x):
        y = np.sum(x) / np.size(x)
        return y

    def corr2(a,b):
        a = a - mean2(a)
        b = b - mean2(b)
        r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum())
        return r

    map1=cv2.resize(saliencyMap1,(saliencyMap2.shape[1],saliencyMap2.shape[0]))
    map1=map1.astype(float)
    map1=map1/255.0
    map2=saliencyMap2.astype(float)
    map2=map2/255.0	
	
    map1=(map1-np.mean(map1))/np.std(map1)
    map2=(map2-np.mean(map2))/np.std(map2)
	
    score=corr2(map1,map2)
	
    return score

def NSS(saliencyMap, fixationMap):
    map=cv2.resize(saliencyMap,(fixationMap.shape[1],fixationMap.shape[0]))
    map=(map-np.mean(map))/np.std(map)
    sum=0
    count=0
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if fixationMap[i][j]!=0:
               sum+=map[i][j]
               count+=1
    score=(float)(sum)/(count)
    return score

def KLdiv(saliencyMap1, saliencyMap2):
    map1=cv2.resize(saliencyMap1,(saliencyMap2.shape[1],saliencyMap2.shape[0]))
    map1=map1.astype(float)
    map1=map1/255.0
    map2=saliencyMap2.astype(float)
    map2=map2/255.0	
	
    map1 = map1/np.sum(map1)
    map2 = map2/np.sum(map2)
	
    score= np.sum(np.sum(map2*np.log(2.2204e-16 + map2/(map1+2.2204e-16))))
	
    return score


def eval_all(img1, img2):
    # AUC_Borji_ = AUC_Borji(img1,img2)
    # AUC_Judd_ = AUC_Judd(img1,img2)
    # AUC_Judd_ = calculate_auc_j(img1,img2)
    CC_ = CC(img1,img2)
    NSS_ = NSS(img1,img2)
    KLdiv_ = KLdiv(img1,img2)
    return  CC_, NSS_, KLdiv_


if __name__ == "__main__":
    img1 = cv2.imread('pic/2.png',cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('pic/1.png',cv2.IMREAD_GRAYSCALE)
    
    print(eval_all(img1, img2))