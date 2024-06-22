import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import glob
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor


class LoadDatasetSkt(Dataset):
    def __init__(self, img_folder_path, skt_folder_path, transform):
        skt_list = []
        label_list = []
        img_name_list = os.listdir(img_folder_path)

        for img_name in img_name_list:
            skt_list_path = glob.glob(skt_folder_path + img_name.split('.')[0] + '_*.png')
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

        return sample_skt, image_idx, skt_path

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


def mian_retrieval(skt_model, img_model, dataset='ShoeV2', mode='test', device='cuda:1'):
    print('Evaluating Network dataset [{}_{}] ...'.format(dataset, mode))

    data_set_skt = LoadDatasetSkt(img_folder_path='./datasets/{}/{}B/'.format(dataset, mode),
                                  skt_folder_path='./datasets/{}/{}A/'.format(dataset, mode),
                                  transform=Compose([Resize(224), ToTensor()]))

    data_set_img = LoadDatasetImg(img_folder_path='./datasets/{}/{}B/'.format(dataset, mode),
                                  skt_folder_path='./datasets/{}/{}A/'.format(dataset, mode),
                                  transform=Compose([Resize(224), ToTensor()]))

    data_loader_skt = DataLoader(data_set_skt, batch_size=10, shuffle=True, num_workers=2, pin_memory=True)
    data_loader_img = DataLoader(data_set_img, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    skt_model = skt_model.to(device)
    img_model = img_model.to(device)
    skt_model.eval()
    img_model.eval()

    img_list = os.listdir('./datasets/{}/{}B/'.format(dataset, mode))

    with torch.no_grad():
        Image_Feature = torch.FloatTensor().to(device)
        for imgs in tqdm(data_loader_img):
            img = imgs.to(device)
            img_feats, _ = img_model(img)
            img_feats = F.normalize(img_feats, dim=1)
            Image_Feature = torch.cat((Image_Feature, img_feats.detach()))

        for idx, skts in enumerate(tqdm(data_loader_skt)):
            skt, skt_idx, target_sketch_paths = skts
            skt, skt_idx = skt.to(device), skt_idx.to(device)
            skt_feats, _ = skt_model(skt)
            skt_feats = F.normalize(skt_feats, dim=1)

            similarity_matrix = torch.argsort(torch.matmul(skt_feats, Image_Feature.T), dim=1, descending=True)
            print('idx: ', idx)
            counts = similarity_matrix[:, 0] == skt_idx
            print(counts)
            for cc, count in enumerate(counts):
                if count == False:
                    print(skt_idx[cc])
                    print(similarity_matrix[cc][:10])

            pred_positions_lists = []

            for pred_idxs in similarity_matrix:
                pred_positions_list = []
                for nums, pred_idx in enumerate(pred_idxs):
                    if nums == 10:
                        break
                    pred_path = os.path.join('./datasets/{}/{}B/'.format(dataset, mode), img_list[pred_idx])
                    pred_positions_list.append(pred_path)

                pred_positions_lists.append(pred_positions_list)

            make_matrix(target_sketch_paths, pred_positions_lists, './SBIR_Chair/{}.png'.format(idx))


def make_matrix(target_sketch_paths, pred_positions_lists, im_name):
    image_matrix = []

    pred_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor()])

    for idx, target_sketch_path in enumerate(target_sketch_paths):
        sketch_img = pred_transform(Image.open(target_sketch_path).convert('RGB'))
        image_matrix.append(sketch_img.cpu())

        for pred_path in pred_positions_lists[idx]:
            pred_img = pred_transform(Image.open(pred_path).convert('RGB'))
            image_matrix.append(pred_img.cpu())

    torchvision.utils.save_image(image_matrix, im_name, nrow=int(len(pred_positions_lists[0]) + 1),
                                 padding=0, normalize=True)

    return im_name


if __name__ == "__main__":
    from train_utils_main import EncoderViT

    checkpoint = torch.load('./results/ChairV2/model_Best.pth')
    print('Loading Pretrained model successful !'
          'Epoch:[{}]  |  Loss:[{}]'.format(checkpoint['epoch'], checkpoint['loss']))
    print('Top1: {} %  |  Top5: {} %  |  Top10: {} %'.format(checkpoint['top1'], checkpoint['top5'],
                                                             checkpoint['top10']))

    sketch_encoder = EncoderViT(num_classes=512, feature_dim=768,
                                encoder_backbone='vit_base_patch16_224')
    image_encoder = EncoderViT(num_classes=512, feature_dim=768,
                               encoder_backbone='vit_base_patch16_224')
    sketch_encoder.load_state_dict(checkpoint['skt_model'])
    image_encoder.load_state_dict(checkpoint['img_model'])

    # mian_retrieval(sketch_encoder, image_encoder, dataset='ChairV2', mode='train')
    mian_retrieval(sketch_encoder, image_encoder, dataset='ChairV2', mode='test')
