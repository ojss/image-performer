import pickle
import os, os.path
import lasagne
import numpy as np
from torch.utils.data import Dataset, DataLoader

class Imagenet64(Dataset):
    def __init__(self, list_IDs,data_folder, img_size = 8):
#         'Initialization'
        self.list_IDs = list_IDs
        self.data_folder = data_folder
        self.img_size = img_size
   
    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo)
        return dict

    def __len__(self):
#         'Denotes the total number of samples'
        return len(self.list_IDs)
    
    def __getitem__(self, index):
#         'Generates one sample of data'
        # Select sample
        img_size = self.img_size
        idx = self.list_IDs[index]
        data_file = os.path.join(self.data_folder, 'train_data_batch_')
#         print('asd',self.list_IDs[index])
        data_file = data_file + str(idx)
        d = self.unpickle(data_file)
        x = d['data']
        y = d['labels']
        mean_image = d['mean']
        
        x = x/np.float32(255)
        mean_image = mean_image/np.float32(255)
    
        # Labels are indexed from 1, shift it so that indexes start at 0
        y = [i-1 for i in y]
        data_size = x.shape[0]
    
        x -= mean_image
    
        img_size2 = img_size * img_size
    
        x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
        x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)
    
        # create mirrored images
        X_train = x[0:data_size, :, :, :]
        Y_train = y[0:data_size]
        X_train_flip = X_train[:, :, :, ::-1]
        Y_train_flip = Y_train
        X_train = np.concatenate((X_train, X_train_flip), axis=0)
        Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)
        X_train=lasagne.utils.floatX(X_train)
        
        return X_train