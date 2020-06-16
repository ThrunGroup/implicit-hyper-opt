import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils


class ISICDataset(Dataset):
    """ISIC dataset."""

    def __init__(self, mdlParams, indSet):
        """
        Args:
            mdlParams (dict): Configuration for loading
            indSet (string): Indicates train, val, test
        """
        # True/False on ordered cropping for eval
        # Copy stuff from config
        self.input_size = (np.int32(mdlParams['input_size'][0]),np.int32(mdlParams['input_size'][1]))        
        self.orderedCrop = mdlParams['orderedCrop']   
        self.balancing = mdlParams['balance_classes']
        self.subtract_set_mean = mdlParams['subtract_set_mean']
        self.same_sized_crop = mdlParams['same_sized_crops']  
        self.train_eval_state = mdlParams['trainSetState']   
        # Potential setMean to deduce from channels
        self.setMean = mdlParams['setMean'].astype(np.float32)
        # Current indSet = 'trainInd'/'valInd'/'testInd'
        self.indices = mdlParams[indSet]  
        self.indSet = indSet
        # Balanced batching
        if self.balancing == 3 and indSet == 'trainInd':
            # Sample classes equally for each batch
            # First, split set by classes
            not_one_hot = np.argmax(mdlParams['labels_array'],1)
            self.class_indices = []
            for i in range(mdlParams['numClasses']):
                self.class_indices.append(np.where(not_one_hot==i)[0])
                # Kick out non-trainind indices
                self.class_indices[i] = np.setdiff1d(self.class_indices[i],mdlParams['valInd'])
            # Now sample indices equally for each batch by repeating all of them to have the same amount as the max number
            indices = []
            max_num = np.max([len(x) for x in self.class_indices])
            # Go thourgh all classes
            for i in range(mdlParams['numClasses']):
                count = 0
                class_count = 0
                max_num_curr_class = len(self.class_indices[i])
                # Add examples until we reach the maximum
                while(count < max_num):
                    # Start at the beginning, if we are through all available examples
                    if class_count == max_num_curr_class:
                        class_count = 0
                    indices.append(self.class_indices[i][class_count])
                    count += 1
                    class_count += 1
            print("Largest class",max_num,"Indices len",len(indices))
            # Set labels/inputs
            self.labels = mdlParams['labels_array'][indices,:]
            self.im_paths = np.array(mdlParams['im_paths'])[indices].tolist()     
            # Normal train proc
            if self.same_sized_crop:
                cropping = transforms.RandomCrop(self.input_size)
            else:
                cropping = transforms.RandomResizedCrop(self.input_size[0])
            # All transforms
            self.composed = transforms.Compose([
                    cropping,
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(brightness=32. / 255.,saturation=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(torch.from_numpy(self.setMean).float(),torch.from_numpy(np.array([1.,1.,1.])).float())
                    ])                                
        elif self.orderedCrop and (indSet == 'valInd' or self.train_eval_state  == 'eval'):
            # Complete labels array, only for current indSet, repeat for multiordercrop
            inds_rep = np.repeat(mdlParams[indSet], mdlParams['multiCropEval'])
            self.labels = mdlParams['labels_array'][inds_rep,:]
            # Path to images for loading, only for current indSet, repeat for multiordercrop
            self.im_paths = np.array(mdlParams['im_paths'])[inds_rep].tolist()
            print(len(self.im_paths))
            # Set up crop positions for every sample
            self.cropPositions = np.tile(mdlParams['cropPositions'], (mdlParams[indSet].shape[0],1))
            print("CP",self.cropPositions.shape)          
            # Set up transforms
            self.norm = transforms.Normalize(torch.from_numpy(self.setMean).float(),torch.from_numpy(np.array([1.,1.,1.])).float())
            self.trans = transforms.ToTensor()
        elif indSet == 'valInd':
            self.cropping = transforms.RandomResizedCrop(self.input_size)
            # Complete labels array, only for current indSet
            self.labels = mdlParams['labels_array'][mdlParams[indSet],:]
            # Path to images for loading, only for current indSet
            self.im_paths = np.array(mdlParams['im_paths'])[mdlParams[indSet]].tolist()            
        else:
            # Normal train proc
            if self.same_sized_crop:
                cropping = transforms.RandomCrop(self.input_size)
            else:
                cropping = transforms.RandomResizedCrop(self.input_size[0])
            # Color distortion
            if mdlParams.get('full_color_distort') is not None:
                color_distort = transforms.ColorJitter(brightness=32. / 255.,saturation=0.5, contrast = 0.5, hue = 0.2) 
            else:
                color_distort = transforms.ColorJitter(brightness=32. / 255.,saturation=0.5) 
            # All transforms
            self.composed = transforms.Compose([
                    cropping,
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    color_distort,
                    transforms.ToTensor(),
                    transforms.Normalize(torch.from_numpy(self.setMean).float(),torch.from_numpy(np.array([1.,1.,1.])).float())
                    ])                  
            # Complete labels array, only for current indSet
            self.labels = mdlParams['labels_array'][mdlParams[indSet],:]
            # Path to images for loading, only for current indSet
            self.im_paths = np.array(mdlParams['im_paths'])[mdlParams[indSet]].tolist()

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        # Load image
        x = Image.open(self.im_paths[idx])
        # Get label
        y = self.labels[idx,:]
        # Transform data based on whether train or not train. If train, also check if its train train or train inference
        if self.orderedCrop and (self.indSet == 'valInd' or self.indSet == 'testInd' or self.train_eval_state == 'eval'):
            # Apply ordered cropping to validation or test set
            # First, to pytorch tensor (0.0-1.0)
            x = self.trans(x)
            # Normalize
            x = self.norm(x)
            # Get current crop position
            x_loc = self.cropPositions[idx,0]
            y_loc = self.cropPositions[idx,1]
            # Then, apply current crop
            x = x[:,(x_loc-np.int32(self.input_size[0]/2.)):(x_loc-np.int32(self.input_size[0]/2.))+self.input_size[0],(y_loc-np.int32(self.input_size[1]/2.)):(y_loc-np.int32(self.input_size[1]/2.))+self.input_size[1]]
        elif self.indSet == 'valInd':
            # First, to pytorch tensor (0.0-1.0)
            x = self.trans(x)
            # Normalize
            x = self.norm(x)
            x = self.cropping(x)
        else:
            # Apply
            x = self.composed(x)  
        # Transform y
        y = np.argmax(y)
        y = np.int64(y)
        return x, y, idx
