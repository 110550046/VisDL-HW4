import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np
from tqdm import tqdm


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
    def __init__(self, channels=128, num_blocks=16):
        super(PromptIR, self).__init__()
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*[PromptBlock(channels) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class RainSnowDataset(Dataset):
    def __init__(self, degraded_dir, clean_dir, transform=None):
        self.degraded_images = sorted(
            [
                os.path.join(degraded_dir, f)
                for f in os.listdir(degraded_dir)
                if f.endswith(".png")
            ]
        )
        self.clean_images = sorted(
            [
                os.path.join(clean_dir, f)
                for f in os.listdir(clean_dir)
                if f.endswith(".png")
            ]
        )
        self.transform = transform

    def __len__(self):
        return len(self.degraded_images)

    def __getitem__(self, idx):
        degraded = Image.open(self.degraded_images[idx]).convert("RGB")
        clean = Image.open(self.clean_images[idx]).convert("RGB")
        if self.transform:
            degraded = self.transform(degraded)
            clean = self.transform(clean)
        return degraded, clean


def psnr_tensor(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def train():
    EPOCHS = 100
    BATCH_SIZE = 8
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_path = "hw4_realse_dataset/train"
    transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )

    degraded_dir = os.path.join(base_path, "degraded")
    clean_dir = os.path.join(base_path, "clean")
    full_dataset = RainSnowDataset(degraded_dir, clean_dir, transform)

    total_len = len(full_dataset)
    val_len = int(0.1 * total_len)
    train_len = total_len - val_len
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_len, val_len]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = PromptIR().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_psnr = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for degraded, clean in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            degraded, clean = degraded.to(DEVICE), clean.to(DEVICE)

            output = model(degraded)
            loss = criterion(output, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(
            f"[Epoch {epoch+1}] Training Loss: {running_loss / len(train_loader):.4f}"
        )

        model.eval()
        val_psnrs = []
        with torch.no_grad():
            for degraded, clean in val_loader:
                degraded = degraded.to(DEVICE)
                clean = clean.to(DEVICE)
                output = model(degraded).clamp(0, 1)
                psnr_val = psnr_tensor(output, clean)
                val_psnrs.append(psnr_val.item())

        avg_psnr = np.mean(val_psnrs)
        print(f"[Epoch {epoch+1}] Validation PSNR: {avg_psnr:.2f}")

        scheduler.step(avg_psnr)

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), "best_promptir.pth")
            print(f"✅ Best model updated: PSNR {avg_psnr:.2f}")


def evaluate(model, transform):
    model.eval()
    test_dir = "hw4_realse_dataset/test/degraded"
    degraded_images = sorted(
        [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".png")]
    )
    psnrs = []

    with torch.no_grad():
        for img_path in degraded_images:
            degraded = Image.open(img_path).convert("RGB")
            degraded_tensor = transform(degraded).unsqueeze(0).cuda()
            output = model(degraded_tensor).clamp(0, 1)
            output_np = output.squeeze().cpu().numpy().transpose(1, 2, 0)

            degraded_np = degraded_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
            psnr = compare_psnr(degraded_np, output_np, data_range=1.0)
            psnrs.append(psnr)

    avg_psnr = np.mean(psnrs)
    print(f"Average PSNR on test set: {avg_psnr:.2f}")
    return avg_psnr


if __name__ == "__main__":
    train()
