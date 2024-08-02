import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LoadMyDataset(Dataset):
    def __init__(self, img_folder_path, skt_folder_path, im_size=224):
        skt_list = []
        img_list = []
        img_name_list = os.listdir(img_folder_path)

        for pos_img_name in img_name_list:
            # 1.1 anchor_skt_name, pos_skt_name
            pos_skt_list_path = glob.glob(skt_folder_path + pos_img_name.split('.')[0] + '_?.png')
            pos_skt_list_name = [file_name.split('/')[-1].split('.')[0] + '.png'
                                 for file_name in pos_skt_list_path]

            # 1.2 append lists
            if len(pos_skt_list_name) == 0:
                print(pos_img_name)

            for len_list in range(len(pos_skt_list_name)):
                skt_list.append(pos_skt_list_name[len_list])
                img_list.append(pos_img_name)

        self.img_folder_path = img_folder_path
        self.skt_folder_path = skt_folder_path

        self.skt_list = skt_list
        self.img_list = img_list

        # A0
        self.transform_anchor = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor()
        ])

        # A1
        self.transform_aug = transforms.Compose([
            transforms.Resize((int(im_size * 1.2), int(im_size * 1.2))),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(p=0.8),
            transforms.CenterCrop((int(im_size), int(im_size))),
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor()
        ])

        # A2
        # self.transform_aug = transforms.Compose([
        #     transforms.RandomResizedCrop(size=(224, 224)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

    def __getitem__(self, item):
        # 1.1 anchor skt
        skt_path = os.path.join(self.skt_folder_path, self.skt_list[item])
        skt_anchor = self.transform_anchor(Image.fromarray(np.array(Image.open(skt_path).convert('RGB'))))
        skt_aug = self.transform_aug(Image.fromarray(np.array(Image.open(skt_path).convert('RGB'))))

        # 1.2 anchor img
        pos_path = os.path.join(self.img_folder_path, self.img_list[item])
        img_anchor = self.transform_anchor(Image.fromarray(np.array(Image.open(pos_path).convert('RGB'))))
        img_aug = self.transform_aug(Image.fromarray(np.array(Image.open(pos_path).convert('RGB'))))

        sample = skt_anchor, skt_aug, img_anchor, img_aug

        return sample

    def __len__(self):
        return len(self.skt_list)


class LoadDatasetSkt(Dataset):
    def __init__(self, img_folder_path, skt_folder_path, transform):
        skt_list = []
        label_list = []
        img_name_list = os.listdir(img_folder_path)

        for img_name in img_name_list:
            skt_list_path = glob.glob(skt_folder_path + img_name.split('.')[0] + '_?.png')
            for skt_name in skt_list_path:
                skt_name = skt_name.split('/')[-1].split('.')[0] + '.png'
                img_item = img_name_list.index(img_name)
                skt_list.append(skt_name)
                label_list.append(img_item)

        self.skt_folder_path = skt_folder_path
        self.transform = transform
        self.skt_list = skt_list
        self.label_list = label_list

    def __getitem__(self, item):
        skt_path = os.path.join(self.skt_folder_path, self.skt_list[item])
        sample_skt = self.transform((Image.fromarray(np.array(Image.open(skt_path).convert('RGB')))))
        image_idx = self.label_list[item]

        return sample_skt, image_idx

    def __len__(self):
        return len(self.skt_list)


class LoadDatasetImg(Dataset):
    def __init__(self, img_folder_path, skt_folder_path, transform):
        self.transform = transform
        self.img_folder_path = img_folder_path
        self.img_list = os.listdir(img_folder_path)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_folder_path, self.img_list[item])
        sample_img = self.transform((Image.fromarray(np.array(Image.open(img_path).convert('RGB')))))

        return sample_img

    def __len__(self):
        return len(self.img_list)
