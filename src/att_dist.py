import torch
import numpy as np
import att_dist_model
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm


def plot_attention_distance(datas, avg, var):
    plot_mean_var(avg, var)

    plt.figure(figsize=(5, 3))
    layers = list(range(1, len(datas) + 1))
    distances = list(range(0, 91, 10))

    colors = ['#FF6A6A', '#ff7f0e', '#32CD32', '#FF0000', '#00CDCD', '#FFD39B',
              '#e377c2', '#B0E2FF', '#bcbd22', '#17becf', '#FFA500', '#A020F0']

    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
    for i in layers:
        for data in datas[i - 1]:
            plt.scatter(i, data, edgecolors='black', s=100, c=colors[i - 1], alpha=0.8)

    plt.xticks(layers)
    plt.yticks(distances)
    plt.xlabel('Model Layers')
    plt.ylabel('Attention Distance')

    plt.show()


def plot_mean_var(avg, var):
    fig, ax1 = plt.subplots(figsize=(4, 3))
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)

    x = list(range(1, 13))
    distances_avg = list(range(0, 91, 10))
    distances_var = list(range(0, 501, 100))

    ax1.plot(x, avg, label='Distance Avg', marker='o', markeredgecolor='black', color='#EE6363', linestyle='--')
    ax1.set_ylabel('Distance Avg')

    ax2 = ax1.twinx()
    ax2.plot(x, var, label='Distance Var', marker='D', markeredgecolor='black', color='#00BFFF', linestyle='--')
    ax2.set_ylabel('Distance Var')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.xticks(x)
    ax1.set_yticks(distances_avg)
    ax2.set_yticks(distances_var)

    plt.show()


def plot_attention(attention_maps):
    attention_probs = nn.Softmax(dim=0)(attention_maps)

    fig, axs = plt.subplots(3, 4, figsize=(15, 9))
    for head_idx in range(12):
        row = head_idx // 4
        col = head_idx % 4

        head_attention_map = attention_probs[head_idx]
        im = axs[row, col].imshow(head_attention_map, cmap='hot', interpolation='nearest')
        axs[row, col].set_title(f"Attention Map - Head {head_idx}")
        axs[row, col].axis('off')

    plt.tight_layout()
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Attention Weight')
    plt.show()


def compute_attention_distance(model, input_data):
    model.eval()
    with torch.no_grad():
        attention_distances_layers = []
        attention_distances_avg = []
        attention_distances_var = []
        attention_maps_layers = model.forward_attn(input_data)
        for idx, attention_maps in enumerate(attention_maps_layers):
            # plot_attention(attention_maps)
            attention_distances = []
            B, H, N, _ = attention_maps.shape
            attention_maps = attention_maps.transpose(0, 1)
            for head in range(H):
                attention = attention_maps[head]
                positions = torch.arange(N).float().unsqueeze(0).to(attention.device)  # (1, N)
                distances = torch.abs(positions - positions.transpose(-1, -2))  # [N, N]
                attention_distance = torch.sum(attention * distances) / (B * N)
                attention_distances.append(attention_distance.item())

            mean_attention_distance = np.mean(attention_distances)
            var_attention_distance = np.var(attention_distances, ddof=1)

            attention_distances_layers.append(attention_distances)
            attention_distances_avg.append(mean_attention_distance)
            attention_distances_var.append(var_attention_distance)

            print('Layer[{}]:'.format(idx))
            print("Attention Distance: ", attention_distances)
            print("Average Attention Distance: ", mean_attention_distance)
            print("Variance Attention Distance: ", var_attention_distance)

    plot_attention_distance(attention_distances_layers, attention_distances_avg, attention_distances_var)


class EncoderViT(nn.Module):
    def __init__(self, num_classes=512, feature_dim=768):
        super().__init__()
        self.encoder = att_dist_model.vit_base_patch16_224(pretrained=True)
        self.num_blocks = 196
        self.mlp_block = nn.Sequential(
            nn.LayerNorm(self.num_blocks),
            nn.Linear(self.num_blocks, 1),
            nn.GELU()
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x):
        x = self.encoder.patch_embed(x)
        cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        if self.encoder.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.encoder.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)
        return x

    def forward_attn(self, x):
        return self.encoder.forward_attn(x)


class LoadDataset(Dataset):
    def __init__(self, img_folder_path):
        self.img_folder_path = img_folder_path
        self.img_name_list = os.listdir(img_folder_path)
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __getitem__(self, item):
        img_path = os.path.join(self.img_folder_path, self.img_name_list[item])
        sample = self.img_transform(Image.open(img_path).convert('RGB'))

        return sample

    def __len__(self):
        return len(self.img_name_list)


if __name__ == '__main__':
    # device = torch.device('cuda:0')
    model = att_dist_model.vit_base_patch16_224(pretrained=True)
    # model = model.to(device)

    encode = EncoderViT()
    # checkpoint = torch.load('./checkpoint/ChairV2/model_Best.pth')
    # checkpoint = torch.load('./checkpoint/ShoeV2/model_Best.pth')
    checkpoint = torch.load('./checkpoint/ClothesV1/model_Best.pth')
    encode.load_state_dict(checkpoint['img_model'])
    # encode.load_state_dict(checkpoint['skt_model'])
    # encode = encode.to(device)

    # data_set = LoadDataset('./datasets/ShoeV2/testB/')
    # data_set = LoadDataset('./datasets/ShoeV2/testA/')
    # data_set = LoadDataset('./datasets/ChairV2/testB/')
    # data_set = LoadDataset('./datasets/ChairV2/testA/')
    data_set = LoadDataset('./datasets/ClothesV1/testB/')
    # data_set = LoadDataset('./datasets/ClothesV1/testA/')
    data_loader = DataLoader(data_set, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)

    for data in tqdm(data_loader):
        # data = data.to(device)
        # compute_attention_distance(model, input_data=data)
        compute_attention_distance(encode, input_data=data)

        break
