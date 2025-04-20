import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from utils import load_image, image_path_city, is_image 
from dataset import id2color, id2trainId 
import torch 

IMAGE_SIZE = [512, 256]

class cityscapes(Dataset):

    def __init__(self, root, transform=None, image_transform=None, subset='train'):
        self.images_root = os.path.join(root, subset)
        print(self.images_root)
        self.filenames = [f for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()
        self.img_transform = image_transform
        self.transform = transform # ADDED THIS


    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')

        image_np, label_np = self.image_mask_split(image_path_city(self.images_root, filename), IMAGE_SIZE)

        if self.img_transform is not None: 
            transformed = self.img_transform(image=image_np)
            image_np = transformed['image']
            
        if self.transform is not None:
            transformed = self.transform(image=image_np, mask=label_np)
            image = transformed['image']
            label_np = transformed['mask']
      
        label = self.find_closest_labels_vectorized_torch(label_np, id2color, id2trainId)
        
        return image, label 

    def __len__(self):
        return len(self.filenames)

    def image_mask_split(self, filename, image_size):
        image_mask = Image.open(filename)
        
        image, mask = image_mask.crop([0, 0, 256, 256]), image_mask.crop([256, 0, 512, 256])
        
        image = image.resize(image_size, Image.BILINEAR)
        mask = mask.resize(image_size, Image.NEAREST)
        
        image = np.array(image, dtype=np.float32) / 255 
        mask = np.array(mask)
        
        return image, mask

    def find_closest_labels_vectorized_torch(self, mask, color_mapping, trainid_mapping):
       
        H, W, C = mask.shape
        mask = mask.float()

        ids = []
        colors = []
        for id, color in color_mapping.items():
            ids.append(id)
            colors.append(torch.tensor(color, dtype=torch.float32))
        colors = torch.stack(colors)
        ids = torch.tensor(ids, dtype=torch.long)

        mask_flat = mask.view(-1, 3)
        colors_exp = colors.view(1, -1, 3)
        dists = torch.norm(mask_flat[:, None] - colors_exp, dim=-1)

        min_dist_indices = torch.argmin(dists, dim=-1)
        closest_ids = ids[min_dist_indices]

        train_ids = torch.full((H * W,), 255, dtype=torch.long)
        for id, train_id in trainid_mapping.items():
            train_ids[closest_ids == id] = train_id
        train_ids = train_ids.view(H, W)

        min_dists = dists[torch.arange(dists.shape[0]), min_dist_indices]
        threshold = 100.0 
        void_mask = min_dists.view(H, W) > threshold
        train_ids[void_mask] = 0
        
        return train_ids
    