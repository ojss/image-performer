import pickle
import os, os.path
import numpy as np
from torch.utils.data import Dataset,TensorDataset, DataLoader, ConcatDataset

class Imagenet64(Dataset):
    def __init__(self, data_folder, filename, img_size = 64):
        self.data_file = os.path.join(data_folder, filename)
#         print(self.data_file)
        self.d = self.unpickle(self.data_file)
        self.img_size = img_size

   
    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo)
        return dict

    def __len__(self):
#         'Denotes the total number of samples'
        return self.d['data'].shape[0]
    
    def __getitem__(self, index):
#         'Generates one sample of data'
        # Select sample
        img_size = self.img_size
        x = self.d['data']

        mean_image = self.d['mean']
        x = x[index]
#         print(x.shape)
        x = x/np.float32(255)
        x -= mean_image
        img_size2 = img_size * img_size
    
        x = np.dstack((x[ :img_size2], x[img_size2:2*img_size2], x[ 2*img_size2:]))
        x = x.reshape(( img_size, img_size, 3)).transpose(2, 0, 1)
        return x
