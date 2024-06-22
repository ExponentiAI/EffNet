"""
Acknowledgements:
1. https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
2. https://github.com/IBM/CrossViT
3. https://github.com/rishikksh20/CrossViT-pytorch
"""

import os
import glob
import random
import numpy as np
from torch import nn
import timm
from timm.models.vision_transformer import VisionTransformer
from timm.models.swin_transformer import SwinTransformer
import torch
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from random import randint


# class EncoderViT(nn.Module):
#     def __init__(self, num_classes=256, embed_dim=1024, encoder_backbone='swin_base_patch4_window7_224'):
#         super().__init__()
#         self.encoder: SwinTransformer = timm.create_model(encoder_backbone, pretrained=True)
#         self.mlp_head = nn.Sequential(
#             nn.Linear(embed_dim, num_classes)
#         )
#
#     def forward(self, x):  # (B, 3, 224, 224)
#         x = self.encoder.patch_embed(x)  # (B, 3136, 128)
#         if self.encoder.absolute_pos_embed is not None:
#             x = x + self.encoder.absolute_pos_embed
#         x = self.encoder.pos_drop(x)
#         x = self.encoder.layers(x)  # (B, 49, 1024)  B L C
#         x = self.encoder.norm(x)
#         x = self.encoder.avgpool(x.transpose(1, 2))  # (B, 1024, 1)  B C 1
#         x = torch.flatten(x, 1)  # (B, 1024)
#         x = self.mlp_head(x)  # (B, cls)
#
#         return x


class EncoderViT(nn.Module):
    def __init__(self, num_classes=256, feature_dim=768, encoder_backbone='vit_base_patch16_224'):
        super().__init__()
        self.encoder: VisionTransformer = timm.create_model(encoder_backbone, pretrained=True)
        self.num_blocks = 196
        self.mlp_block = nn.Sequential(
            nn.LayerNorm(self.num_blocks),
            nn.Linear(self.num_blocks, 1),
            nn.GELU()
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(feature_dim, num_classes)
        )

        # self.alpha = nn.Parameter(torch.tensor([1.]))

    def embedding(self, image):
        x = self.encoder.patch_embed(image)
        cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        if self.encoder.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.encoder.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)
        return x

    def forward(self, image):
        vit_feat = self.embedding(image)
        mlp_feat = self.mlp_head(vit_feat[:, 0])
        return mlp_feat, vit_feat


class CrossViT(nn.Module):
    def __init__(self, num_classes=256, feature_dim=768, cross_heads=[12, 12, 12]):
        super().__init__()
        self.cross_blocks = nn.ModuleList([
            CrossFeature(feature_dim=feature_dim, num_heads=cross_heads[i], qkv_bias=True)
            for i in range(len(cross_heads))])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Dropout(),
            nn.Linear(feature_dim * 2, num_classes),
            nn.GELU(),
            nn.LayerNorm(num_classes),
            nn.Linear(num_classes, num_classes)
        )

    def forward_feature(self, feature_1, feature_2):
        for cross_block in self.cross_blocks:
            feature_1, feature_2 = cross_block(feature_1, feature_2)
        return feature_1, feature_2

    def forward(self, feature_1, feature_2):
        feature_1, feature_2 = self.forward_feature(feature_1, feature_2)

        return self.mlp_head(torch.cat((torch.mean(feature_1, dim=1),
                                        torch.mean(feature_2, dim=1)), dim=1))


class CrossFeature(nn.Module):
    def __init__(self, feature_dim=768, num_heads=12, qkv_bias=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(feature_dim)
        self.attn1 = CrossAttention(feature_dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.mlp1 = nn.Sequential(nn.LayerNorm(feature_dim),
                                  nn.Linear(feature_dim, feature_dim * 2),
                                  nn.GELU(),
                                  nn.Linear(feature_dim * 2, feature_dim))

        self.norm2 = nn.LayerNorm(feature_dim)
        self.attn2 = CrossAttention(feature_dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.mlp2 = nn.Sequential(nn.LayerNorm(feature_dim),
                                  nn.Linear(feature_dim, feature_dim * 2),
                                  nn.GELU(),
                                  nn.Linear(feature_dim * 2, feature_dim))

    def forward(self, feat_1, feat_2):
        # cross attention for feat_1 and feat_2
        cal_qkv = torch.cat((feat_2[:, 0:1], feat_1), dim=1)
        cal_out = feat_1 + self.attn1(self.norm1(cal_qkv))
        feature_1 = cal_out + self.mlp1(cal_out)

        cal_qkv = torch.cat((feat_1[:, 0:1], feat_2), dim=1)
        cal_out = feat_2 + self.attn2(self.norm2(cal_qkv))
        feature_2 = cal_out + self.mlp2(cal_out)

        return feature_1, feature_2


class CrossAttention(nn.Module):
    def __init__(self, feature_dim=768, num_heads=12, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.wq = nn.Linear(feature_dim, feature_dim, bias=qkv_bias)
        self.wk = nn.Linear(feature_dim, feature_dim, bias=qkv_bias)
        self.wv = nn.Linear(feature_dim, feature_dim, bias=qkv_bias)
        self.linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(x[:, 1:, ...]).reshape(B, N - 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x[:, 1:, ...]).reshape(B, N - 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)  # BH1(C/H) @ BH(C/H)N -> BH1N
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C

        return self.linear(x)


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print('Total params: ', total_num, '\nTrainable params: ', trainable_num)


class LoadMyDataset(Dataset):
    def __init__(self, img_folder_path, skt_folder_path, im_size=224):
        self.img_folder_path = img_folder_path
        self.skt_folder_path = skt_folder_path

        self.skt_list = os.listdir(skt_folder_path)
        self.img_list = os.listdir(img_folder_path)

        self.transform_anchor = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor()
        ])

        # self.transform_anchor = transforms.Compose([
        #     transforms.Resize((int(im_size * 1.2), int(im_size * 1.2))),
        #     transforms.RandomRotation(30),
        #     transforms.RandomHorizontalFlip(p=0.8),
        #     transforms.CenterCrop((int(im_size), int(im_size))),
        #     transforms.Resize((im_size, im_size)),
        #     transforms.ToTensor()
        # ])

        # self.transform_anchor = transforms.Compose([
        #     transforms.RandomResizedCrop(size=(224, 224)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

    def __getitem__(self, item):
        anchor_name = self.skt_list[item]
        anchor_path = os.path.join(self.skt_folder_path, anchor_name)

        pos_name = '_'.join(anchor_name.split('_')[: -1]) + '.png'
        pos_path = os.path.join(self.img_folder_path, pos_name)

        neg_img_list = [x for x in self.img_list if x != pos_name]
        neg_list = list(range(len(neg_img_list)))
        neg_item = neg_list[randint(0, len(neg_list) - 1)]
        neg_path = os.path.join(self.img_folder_path, neg_img_list[neg_item])

        # 1.1 anchor skt
        anchor = self.transform_anchor(Image.open(anchor_path).convert('RGB'))
        pos = self.transform_anchor(Image.open(pos_path).convert('RGB'))
        neg = self.transform_anchor(Image.open(neg_path).convert('RGB'))

        sample = anchor, pos, neg

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
                skt_name = skt_name.replace("\\", "/").split('/')[-1].split('.')[0] + '.png'
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


def get_acc(skt_model, img_model, batch_size=128, dataset='ClothesV1', mode='test', device='cuda'):
    print('Evaluating Network dataset [{}_{}] ...'.format(dataset, mode))

    data_set_skt = LoadDatasetSkt(img_folder_path='./datasets/{}/{}B/'.format(dataset, mode),
                                  skt_folder_path='./datasets/{}/{}A/'.format(dataset, mode),
                                  transform=Compose([Resize(224), ToTensor()]))

    data_set_img = LoadDatasetImg(img_folder_path='./datasets/{}/{}B/'.format(dataset, mode),
                                  skt_folder_path='./datasets/{}/{}A/'.format(dataset, mode),
                                  transform=Compose([Resize(224), ToTensor()]))

    data_loader_skt = DataLoader(data_set_skt, batch_size=batch_size,
                                 shuffle=True, num_workers=2, pin_memory=True)
    data_loader_img = DataLoader(data_set_img, batch_size=batch_size,
                                 shuffle=False, num_workers=2, pin_memory=True)

    skt_model = skt_model.to(device)
    img_model = img_model.to(device)
    skt_model.eval()
    img_model.eval()

    top1_count = 0
    top5_count = 0
    top10_count = 0

    with torch.no_grad():
        Image_Feature = torch.FloatTensor().to(device)
        for imgs in tqdm(data_loader_img):
            img = imgs.to(device)
            img_feats, _ = img_model(img)
            img_feats = F.normalize(img_feats, dim=1)
            Image_Feature = torch.cat((Image_Feature, img_feats.detach()))

        for idx, skts in enumerate(tqdm(data_loader_skt)):
            skt, skt_idx = skts
            skt, skt_idx = skt.to(device), skt_idx.to(device)
            skt_feats, _ = skt_model(skt)
            skt_feats = F.normalize(skt_feats, dim=1)

            similarity_matrix = torch.argsort(torch.matmul(skt_feats, Image_Feature.T), dim=1, descending=True)

            top1_count += (similarity_matrix[:, 0] == skt_idx).sum()
            top5_count += (similarity_matrix[:, :5] == torch.unsqueeze(skt_idx, dim=1)).sum()
            top10_count += (similarity_matrix[:, :10] == torch.unsqueeze(skt_idx, dim=1)).sum()

        top1_accuracy = round(top1_count.item() / len(data_set_skt) * 100, 3)
        top5_accuracy = round(top5_count.item() / len(data_set_skt) * 100, 3)
        top10_accuracy = round(top10_count.item() / len(data_set_skt) * 100, 3)

    return top1_accuracy, top5_accuracy, top10_accuracy


#  InfoNCE Loss
def cross_loss(feature_1, feature_2, args):
    labels = torch.cat([torch.arange(len(feature_1)) for _ in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(args.device)

    # normalize
    feature_1 = F.normalize(feature_1, dim=1)
    feature_2 = F.normalize(feature_2, dim=1)
    features = torch.cat((feature_1, feature_2), dim=0)  # (2*B, Feat_dim)

    similarity_matrix = torch.matmul(features, features.T)  # (2*B, 2*B)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(args.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)  # (2*B, 2*B - 1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # (2*B, 1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # (2*B, 2*(B - 1))

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(args.device)

    logits = logits / args.temperature

    return nn.CrossEntropyLoss()(logits, labels)


if __name__ == '__main__':
    encoder_1 = EncoderViT(num_classes=256, feature_dim=768, encoder_backbone='vit_base_patch16_224')
    encoder_2 = CrossViT(num_classes=256, feature_dim=768, cross_heads=[12, 12, 12])
    get_parameter_number(encoder_1)
    get_parameter_number(encoder_2)
    # Total params: 87,423,541
    # Trainable params: 87,423,541
    # Total params: 38,279,168
    # Trainable params: 38,279,168

    img = torch.randn((1, 3, 224, 224))
    out1, out2 = encoder_1(img)
    out = encoder_2(out2, out2)

    print('Done !')
