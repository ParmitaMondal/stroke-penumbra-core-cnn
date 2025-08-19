import os, argparse, glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

class StrokeSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=256):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
        self.mask_paths = [os.path.join(mask_dir, os.path.basename(p)) for p in self.img_paths]
        self.tf_img = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.size = size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img = Image.open(self.img_paths[i]).convert("RGB")
        msk = Image.open(self.mask_paths[i]).convert("L")  # 0,1,2

        img = self.tf_img(img)

        msk = msk.resize((self.size, self.size), resample=Image.NEAREST)
        msk = torch.from_numpy(np.array(msk, dtype=np.int64))

        return img, msk

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
    )

class TinyUNet(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.enc1 = conv_block(1, 16)
        self.enc2 = conv_block(16, 32)
        self.pool = nn.MaxPool2d(2)

        self.bott = conv_block(32, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = conv_block(64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = conv_block(32, 16)

        self.head = nn.Conv2d(16, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)            # (B,16,H,W)
        p1 = self.pool(e1)           # (B,16,H/2,W/2)
        e2 = self.enc2(p1)           # (B,32,H/2,W/2)
        p2 = self.pool(e2)           # (B,32,H/4,W/4)

        b  = self.bott(p2)           # (B,64,H/4,W/4)

        u2 = self.up2(b)             # (B,32,H/2,W/2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)            # (B,16,H,W)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        logits = self.head(d1)       # (B,3,H,W)
        return logits


def dice_per_class(logits, target, eps=1e-6):
    # logits: (B,C,H,W), target: (B,H,W) in {0..C-1}
    preds = torch.argmax(logits, dim=1)
    dices = []
    C = logits.shape[1]
    for c in range(C):
        p = (preds == c).float()
        t = (target == c).float()
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        dice = (2 * inter + eps) / (denom + eps)
        dices.append(dice.item())
    return dices  # list of length C

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=str, required=True)
    ap.add_argument("--masks", type=str, required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="checkpoints")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = StrokeSegDataset(args.images, args.masks, size=args.size)
    n_total = len(ds)
    n_val = max(1, int(0.2 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    vl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model = TinyUNet(n_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()  # simple; adjust class weights if needed
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(tl, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        running_loss = 0.0
        for imgs, msk in pbar:
            imgs, msk = imgs.to(device), msk.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, msk)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # validation
        model.eval()
        dices_all = []
        with torch.no_grad():
            for imgs, msk in vl:
                imgs, msk = imgs.to(device), msk.to(device)
                logits = model(imgs)
                dices = dice_per_class(logits, msk)  # [bg, core, penumbra]
                dices_all.append(dices)
        if dices_all:
            dices_mean = np.array(dices_all).mean(axis=0)
            print(f"Val Dice - bg:{dices_mean[0]:.3f} core:{dices_mean[1]:.3f} pen:{dices_mean[2]:.3f}")
            mean_foreground = dices_mean[1:].mean()
            if mean_foreground > best_val:
                best_val = mean_foreground
                ckpt = os.path.join(args.out, f"tiny_unet_best_{best_val:.3f}.pth")
                torch.save(model.state_dict(), ckpt)
                print(f"Saved {ckpt}")

    final = os.path.join(args.out, "tiny_unet_final.pth")
    torch.save(model.state_dict(), final)
    print(f"Saved {final}")

if __name__ == "__main__":
    main()
