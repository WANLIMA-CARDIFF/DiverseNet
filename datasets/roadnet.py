import os.path
import random
import cv2
import glob
import numpy as np
from PIL import Image
from torch.utils.data import  Dataset
import torchvision.transforms
from .utils import MaskToTensor, get_params, affine_transform, convert_from_color_annotation

class RoadNet(Dataset):
    """A dataset class for road dataset.
    """
    colormap = [
        (255, 255, 255),
        (0, 0, 255),
    ]
    def __init__(self, data_root, transforms=None, split="train"):
       
        self.img_paths = glob.glob(os.path.join(data_root, '{}_image'.format(split), '*.png'))
        self.segment_dir = os.path.join(data_root, '{}_segment'.format(split))
        self.edge_dir = os.path.join(data_root, '{}_edge'.format(split))
        self.centerline_dir = os.path.join(data_root, '{}_centerline'.format(split))

        self.img_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                       (0.5, 0.5, 0.5))])
        self.rgb_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.lab_transform = MaskToTensor()

    def __getitem__(self, index):
      
        # read a image given a random integer index
        img_path = self.img_paths[index]
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # set paths of annotation maps
        segment_path = os.path.join(self.segment_dir, os.path.basename(img_path))
        edge_path = os.path.join(self.edge_dir, os.path.basename(img_path))
        centerline_path = os.path.join(self.centerline_dir, os.path.basename(img_path))

        # load annotation maps and only use the red channel
        segment = cv2.imread(segment_path, cv2.IMREAD_UNCHANGED)
        edge = cv2.imread(edge_path, cv2.IMREAD_UNCHANGED)
        centerline = cv2.imread(centerline_path, cv2.IMREAD_UNCHANGED)

        # from color to gray
        segment = convert_from_color_annotation(segment)
        edge = convert_from_color_annotation(edge)
        centerline = convert_from_color_annotation(centerline)

        # binarize annotation maps
        _, segment = cv2.threshold(segment, 127, 1, cv2.THRESH_BINARY)
        _, edge = cv2.threshold(edge, 127, 1, cv2.THRESH_BINARY)
        _, centerline = cv2.threshold(centerline, 127, 1, cv2.THRESH_BINARY)

        # apply the transform to both A and B
        image_ = self.img_transforms(Image.fromarray(image.copy()))
        rgb = self.rgb_transforms(Image.fromarray(image.copy()))
        segment = self.lab_transform(segment.copy())
        edge = self.lab_transform(edge.copy()).unsqueeze(0).float()
        centerline = self.lab_transform(centerline.copy()).unsqueeze(0).float()

        return {'image': image_,
                'rgb': rgb,
                'label': segment,
                'edge': edge,
                'centerline': centerline,
                'sample_name': img_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_paths)
        
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        cmap = np.zeros((256, 3), dtype='uint8')
        for i, c in enumerate(cls.colormap):
            cmap[i]=np.array(list(c))
        return cmap[mask]
