"""
Acknowledgements:
1. https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
2. https://github.com/IBM/CrossViT
3. https://github.com/rishikksh20/CrossViT-pytorch
"""

import timm
from timm.models.vision_transformer import VisionTransformer
import torch
from torch import nn


class EncoderViT(nn.Module):
    def __init__(self, num_classes=256, feature_dim=768, encoder_backbone='vit_base_patch16_224'):
        super().__init__()
        self.encoder: VisionTransformer = timm.create_model(encoder_backbone, pretrained=True)

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


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print('Total params: ', total_num, '\nTrainable params: ', trainable_num)


if __name__ == '__main__':
    encoder = EncoderViTPlus(ckpt_path='./checkpoint/ChairV2/model_Best.pth')
    get_parameter_number(encoder)

    img = torch.randn((1, 3, 224, 224))
    skt = torch.randn((1, 3, 224, 224))
    out1, out2 = encoder(img, skt)

    print('Done !')
