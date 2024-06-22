import torch.nn as nn
import torchvision.models as backbone_
import torch.nn.functional as F
import torch


class Backbone_Inception(nn.Module):
    def __init__(self):
        super(Backbone_Inception, self).__init__()
        backbone = backbone_.inception_v3(pretrained=True)

        # Extract Inception Layers #
        self.Conv2d_1a_3x3 = backbone.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = backbone.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = backbone.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = backbone.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = backbone.Conv2d_4a_3x3
        self.Mixed_5b = backbone.Mixed_5b
        self.Mixed_5c = backbone.Mixed_5c
        self.Mixed_5d = backbone.Mixed_5d
        self.Mixed_6a = backbone.Mixed_6a
        self.Mixed_6b = backbone.Mixed_6b
        self.Mixed_6c = backbone.Mixed_6c
        self.Mixed_6d = backbone.Mixed_6d
        self.Mixed_6e = backbone.Mixed_6e

        self.Mixed_7a = backbone.Mixed_7a
        self.Mixed_7b = backbone.Mixed_7b
        self.Mixed_7c = backbone.Mixed_7c

        self.pool_method = nn.AdaptiveMaxPool2d(1)

        self.linear_out = nn.Sequential(
            nn.Linear(2048, 512))

    def embedding(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        backbone_tensor = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        feature = self.pool_method(backbone_tensor)
        feature = torch.flatten(feature, 1)

        return feature

    def forward(self, x):
        feature = self.embedding(x)
        out = self.linear_out(feature)

        return out, feature


class Backbone_Resnet50(nn.Module):
    def __init__(self):
        super(Backbone_Resnet50, self).__init__()
        backbone = backbone_.resnet50(pretrained=True)

        self.features = nn.Sequential()
        for name, module in backbone.named_children():
            if name not in ['avgpool', 'fc']:
                self.features.add_module(name, module)
        self.pool_method = nn.AdaptiveMaxPool2d(1)
        self.output_layer = nn.Linear(2048, 512)

    def embedding(self, x):
        x = self.features(x)
        feature = self.pool_method(x)
        feature = torch.flatten(feature, 1)

        return feature

    def forward(self, x):
        feature = self.embedding(x)
        out = self.output_layer(feature)
        return out, feature


class Backbone_VGG16(nn.Module):
    def __init__(self):
        super(Backbone_VGG16, self).__init__()
        self.backbone = backbone_.vgg16(pretrained=True).features
        self.pool_method = nn.AdaptiveMaxPool2d(1)
        self.output_layer = nn.Linear(512, 512)

    def embedding(self, x):
        y = self.backbone(x)
        feature = self.pool_method(y).view(-1, 512)

        return feature

    def forward(self, x):
        feature = self.embedding(x)
        out = self.output_layer(feature)
        return out, feature


if __name__ == '__main__':
    model = Backbone_VGG16()
    # model = Backbone_Resnet50()
    # model = Backbone_Inception()
    x = torch.randn((1, 3, 224, 224))
    y = model(x)
