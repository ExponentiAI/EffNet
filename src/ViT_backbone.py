import timm
from timm.models.vision_transformer import VisionTransformer
from timm.models.swin_transformer import SwinTransformer
import torch
from torch import nn


# Swin-ViT
class EncoderSViT(nn.Module):
    def __init__(self, num_classes=512, embed_dim=1024, encoder_backbone='swin_base_patch4_window7_224'):
        super().__init__()
        self.encoder: SwinTransformer = timm.create_model(encoder_backbone, pretrained=True)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, num_classes)
        )

    def embedding(self, x):
        x = self.encoder.patch_embed(x)  # (B, 3136, 128)
        if self.encoder.absolute_pos_embed is not None:
            x = x + self.encoder.absolute_pos_embed
        x = self.encoder.pos_drop(x)
        x = self.encoder.layers(x)  # (B, 49, 1024)  B L C
        x = self.encoder.norm(x)
        x = self.encoder.avgpool(x.transpose(1, 2))  # (B, 1024, 1)  B C 1
        x = torch.flatten(x, 1)  # (B, 1024)

        return x

    def forward(self, x):  # (B, 3, 224, 224)
        x = self.embedding(x)
        cls = self.mlp_head(x)  # (B, cls)

        return cls, x


# ViT
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


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print('Total params: ', total_num, '\nTrainable params: ', trainable_num)


if __name__ == '__main__':
    encoder = EncoderViT(num_classes=256, feature_dim=768, encoder_backbone='vit_base_patch16_224')
    # encoder = EncoderViT(num_classes=512, encoder_backbone='swin_base_patch4_window7_224')
    get_parameter_number(encoder)

    img = torch.randn((1, 3, 224, 224))
    out1, out2 = encoder(img)

    print('Done !')
