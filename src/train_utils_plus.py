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
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision import transforms


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


class EncoderViTPlus(nn.Module):
    def __init__(self, ckpt_path='./checkpoint/ChairV2/model_Best.pth'):
        super().__init__()
        self.img_model = EncoderViT(num_classes=512, feature_dim=768, encoder_backbone='vit_base_patch16_224')
        self.skt_model = EncoderViT(num_classes=512, feature_dim=768, encoder_backbone='vit_base_patch16_224')
        checkpoint = torch.load(ckpt_path)
        print('Loading Pretrained model successful !'
              'Epoch:[{}]  |  Loss:[{}]'.format(checkpoint['epoch'], checkpoint['loss']))
        print('Top1: {} %  |  Top5: {} %  |  Top10: {} %'.format(checkpoint['top1'], checkpoint['top5'],
                                                                 checkpoint['top10']))
        self.img_model.load_state_dict(checkpoint['img_model'])
        self.skt_model.load_state_dict(checkpoint['skt_model'])

        for param in self.img_model.parameters():
            param.requires_grad = False
        for param in self.skt_model.parameters():
            param.requires_grad = False

        self.img_model.eval()
        self.skt_model.eval()

        self.trsm = TRSM(feature_dim=768, num_heads=16, qkv_bias=True, drop_rate=0.2)

        self.img_mlp_head = nn.Linear(768, 512)
        self.skt_mlp_head = nn.Linear(768, 512)

    def forward(self, image, sketch):
        vit_img_feat = self.img_model.embedding(image)
        vit_skt_feat = self.skt_model.embedding(sketch)

        vit_img_feat, vit_skt_feat = self.trsm(vit_img_feat, vit_skt_feat)

        # mlp_img_feat = self.img_model.mlp_head(vit_img_feat)
        # mlp_skt_feat = self.skt_model.mlp_head(vit_skt_feat)

        mlp_img_feat = self.img_mlp_head(vit_img_feat)
        mlp_skt_feat = self.skt_mlp_head(vit_skt_feat)

        return mlp_img_feat, vit_img_feat, mlp_skt_feat, vit_skt_feat

    def embedding(self, image, sketch):
        vit_img_feat = self.img_model.embedding(image)
        vit_skt_feat = self.skt_model.embedding(sketch)

        vit_img_feat, vit_skt_feat = self.trsm(vit_img_feat, vit_skt_feat)

        # mlp_img_feat = self.img_model.mlp_head(vit_img_feat)
        # mlp_skt_feat = self.skt_model.mlp_head(vit_skt_feat)

        mlp_img_feat = self.img_mlp_head(vit_img_feat)
        mlp_skt_feat = self.skt_mlp_head(vit_skt_feat)

        return mlp_img_feat, mlp_skt_feat


class TRSM(nn.Module):
    def __init__(self, feature_dim=768, num_heads=16, qkv_bias=True, drop_rate=0.1):
        super().__init__()
        self.in_attn = IntraAttention(feature_dim, num_heads, qkv_bias, drop_rate)
        self.inter_attn = InterAttention(feature_dim, num_heads, qkv_bias, drop_rate)

        self.proj1 = nn.Linear(feature_dim * 2, feature_dim)
        self.proj2 = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, x1, x2):
        in_x1, in_x2 = self.in_attn(x1, x2)
        inter_x1, inter_x2 = self.inter_attn(x1, x2)

        x1 = self.proj1(torch.cat([in_x1, inter_x1], dim=-1))
        x2 = self.proj2(torch.cat([in_x2, inter_x2], dim=-1))

        return x1[:, 0], x2[:, 0]


class QKVAttention(nn.Module):
    def __init__(self, feature_dim=768, num_heads=16, qkv_bias=True, drop_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = feature_dim // num_heads
        self.scale = head_dim ** -0.5

        self.wq = nn.Linear(feature_dim, feature_dim, bias=qkv_bias)
        self.wk = nn.Linear(feature_dim, feature_dim, bias=qkv_bias)
        self.wv = nn.Linear(feature_dim, feature_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(drop_rate)

        self.linear = nn.Linear(feature_dim, feature_dim)
        self.linear_drop = nn.Dropout(drop_rate)

    def forward(self, x):
        B, N, C = x.shape
        q = self.wq(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH N(C/H) @ BH(C/H)N -> BH NN
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        y = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (BH NN @ BHN(C/H)) -> BH N(C/H) -> BN H(C/H) -> BNC
        y = y + x

        out = self.linear(y)
        out = self.linear_drop(out)
        out = out + y

        return out + x


class IntraAttention(nn.Module):
    def __init__(self, feature_dim=768, num_heads=16, qkv_bias=True, drop_rate=0.1):
        super().__init__()
        self.feat1_attn = QKVAttention(feature_dim, num_heads, qkv_bias, drop_rate)
        self.feat2_attn = QKVAttention(feature_dim, num_heads, qkv_bias, drop_rate)

    def forward(self, x1, x2):
        x1 = self.feat1_attn(x1)
        x2 = self.feat1_attn(x2)

        return x1, x2


class InterAttention(nn.Module):
    def __init__(self, feature_dim=768, num_heads=16, qkv_bias=True, drop_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = feature_dim // num_heads
        self.scale = head_dim ** -0.5

        self.wq = nn.Linear(feature_dim, feature_dim, bias=qkv_bias)
        self.wk = nn.Linear(feature_dim, feature_dim, bias=qkv_bias)
        self.wv1 = nn.Linear(feature_dim, feature_dim, bias=qkv_bias)
        self.wv2 = nn.Linear(feature_dim, feature_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(drop_rate)

        self.linear1 = nn.Linear(feature_dim, feature_dim)
        self.linear2 = nn.Linear(feature_dim, feature_dim)
        self.linear1_drop = nn.Dropout(drop_rate)
        self.linear2_drop = nn.Dropout(drop_rate)

    def forward(self, x1, x2):
        assert x1.shape == x2.shape, 'the shape of sketch and image must be same !'
        B, N, C = x1.shape
        q = self.wq(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(x2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v1 = self.wv1(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v2 = self.wv2(x2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH N(C/H) @ BH(C/H)N -> BH NN
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        y1 = (attn @ v1).transpose(1, 2).reshape(B, N, C)  # (BH NN @ BHN(C/H)) -> BH N(C/H) -> BN H(C/H) -> BNC
        y1 = y1 + x1
        out1 = self.linear1(y1)
        out1 = self.linear1_drop(out1)
        out1 = out1 + y1

        y2 = (attn @ v2).transpose(1, 2).reshape(B, N, C)  # (BH NN @ BHN(C/H)) -> BH N(C/H) -> BN H(C/H) -> BNC
        y2 = y2 + x2
        out2 = self.linear2(y2)
        out2 = self.linear2_drop(out2)
        out2 = out2 + y2

        return out1 + x1, out2 + x2


###############################
#       Cross Attention       #
###############################
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
        skt_list = []
        img_list = []
        # skt_pos_list = []
        img_name_list = os.listdir(img_folder_path)

        for pos_img_name in img_name_list:
            # 1.1 anchor_skt_name, pos_skt_name
            # pos_skt_list_path = glob.glob(skt_folder_path + pos_img_name.split('.')[0] + '_?.png')
            pos_skt_list_path = glob.glob(skt_folder_path + pos_img_name.split('.')[0] + '_*.png')
            pos_skt_list_name = [file_name.replace("\\", "/").split('/')[-1].split('.')[0] + '.png'
                                 for file_name in pos_skt_list_path]

            # 1.3 append lists
            if len(pos_skt_list_name) == 0:
                print(pos_img_name)

            for len_list in range(len(pos_skt_list_name)):
                skt_list.append(pos_skt_list_name[len_list])
                img_list.append(pos_img_name)

            # if len(pos_skt_list_name) == 0:
            #     continue
            #
            # elif len(pos_skt_list_name) == 1:
            #     skt_list.append(pos_skt_list_name[0])
            #     img_list.append(pos_img_name)
            #     skt_pos_list.append(pos_skt_list_name[0])
            #
            # else:
            #     random_idx = random.randint(1, len(pos_skt_list_name) - 1)
            #     skt_list.append(pos_skt_list_name[0])
            #     img_list.append(pos_img_name)
            #     skt_pos_list.append(pos_skt_list_name[random_idx])

        self.img_folder_path = img_folder_path
        self.skt_folder_path = skt_folder_path

        self.skt_list = skt_list
        self.img_list = img_list
        # self.skt_pos_list = skt_pos_list

        self.transform_anchor = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor()
        ])

        self.transform_aug = transforms.Compose([
            transforms.Resize((int(im_size * 1.2), int(im_size * 1.2))),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(p=0.8),
            transforms.CenterCrop((int(im_size), int(im_size))),
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor()
        ])

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


def get_acc(model, batch_size=128, dataset='ClothesV1', mode='test', device='cuda'):
    print('Evaluating Network dataset [{}_{}] ...'.format(dataset, mode))

    data_set_skt = LoadDatasetSkt(img_folder_path='./datasets/{}/{}B/'.format(dataset, mode),
                                  skt_folder_path='./datasets/{}/{}A/'.format(dataset, mode),
                                  transform=Compose([Resize(224), ToTensor()]))

    data_set_img = LoadDatasetImg(img_folder_path='./datasets/{}/{}B/'.format(dataset, mode),
                                  skt_folder_path='./datasets/{}/{}A/'.format(dataset, mode),
                                  transform=Compose([Resize(224), ToTensor()]))

    data_loader_skt = DataLoader(data_set_skt, batch_size=len(data_set_img),
                                 shuffle=True, num_workers=4, pin_memory=True)
    data_loader_img = DataLoader(data_set_img, batch_size=len(data_set_img),
                                 shuffle=False, num_workers=0, pin_memory=True)

    model = model.to(device)
    model.eval()

    top1_count = 0
    top5_count = 0
    top10_count = 0

    with torch.no_grad():
        for img in data_loader_img:
            img = img.to(device)
            for skt, skt_idx in tqdm(data_loader_skt):
                if skt.shape[0] != img.shape[0]:
                    filled_tensor = torch.zeros(img.shape)
                    filled_tensor[:skt.shape[0], ...] = skt
                    skt = filled_tensor

                skt, skt_idx = skt.to(device), skt_idx.to(device)

                img_feats, skt_feats = model.embedding(img, skt)
                img_feats = F.normalize(img_feats, dim=1)
                skt_feats = F.normalize(skt_feats, dim=1)

                similarity_matrix = torch.argsort(torch.matmul(skt_feats, img_feats.T), dim=1, descending=True)
                similarity_matrix = similarity_matrix[:skt_idx.shape[0], :]

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
    out1, out2 = encoder_1.forward_feature(img)
    out = encoder_2(out2, out2)

    print('Done !')
