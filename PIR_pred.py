import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class PromptBlock(nn.Module):
    def __init__(self, channels):
        super(PromptBlock, self).__init__()
        self.prompt = nn.Parameter(torch.randn(1, channels, 1, 1))
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x + self.prompt
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class PromptIR(nn.Module):
    def __init__(self, channels=64, num_blocks=8):
        super(PromptIR, self).__init__()
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*[PromptBlock(channels) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


def predict_and_save():
    # Paths
    model_path = "best_promptir.pth"
    test_folder = "hw4_realse_dataset/test/degraded"
    output_npz = "PIR_pred.npz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PromptIR()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.ToTensor()

    images_dict = {}

    with torch.no_grad():
        for filename in sorted(os.listdir(test_folder)):
            if not filename.endswith(".png"):
                continue

            filepath = os.path.join(test_folder, filename)
            image = Image.open(filepath).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).to(device)

            output = model(img_tensor).clamp(0, 1)
            output_np = output.squeeze().cpu().numpy()
            output_uint8 = (output_np * 255).round().astype(np.uint8)

            images_dict[filename] = output_uint8

    np.savez(output_npz, **images_dict)
    print(f"✅ Saved {len(images_dict)} images to {output_npz}")


if __name__ == "__main__":
    predict_and_save()
