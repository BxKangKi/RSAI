#!/usr/bin/env python3
"""
Cloud Removal U-Net: Single executable file
Run: python unet_cloud_removal.py --mode train (or preprocess, test, merge)
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# U-Net MODEL
# =============================================================================
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        
        d2 = self.upconv3(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv2(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.final_conv(d1))

# =============================================================================
# DATASET & PREPROCESS
# =============================================================================
def split_images_to_patches(input_dir, output_dir, patch_size=256):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    valid_patches = []
    index = 0
    
    for img_path in input_dir.glob("*.png"):
        img = Image.open(img_path).convert('RGB')
        W, H = img.size
        
        for y in range(0, H, patch_size):
            for x in range(0, W, patch_size):
                if x + patch_size <= W and y + patch_size <= H:
                    patch = img.crop((x, y, x+patch_size, y+patch_size))
                    patch_name = f"{index}_y{y//patch_size}_x{x//patch_size}.png"
                    patch_path = output_dir / patch_name
                    patch.save(patch_path)
                    valid_patches.append(patch_name)
        index += 1
        
        print(f"Split {img_path.name} → {len([p for p in valid_patches if str(img_path.stem) in p])} patches")
    
    print(f"Total {len(valid_patches)} valid 256x256 patches saved to {output_dir}")
    return valid_patches

class PatchDataset(Dataset):
    def __init__(self, cloudy_dir, clear_dir, transform=None, zero_thr=0.8):
        self.cloudy_dir = Path(cloudy_dir)
        self.clear_dir = Path(clear_dir)
        self.transform = transform
        self.zero_thr = zero_thr

        cloudy_files = sorted(self.cloudy_dir.glob("*.png"))
        clear_files = sorted(self.clear_dir.glob("*.png"))

        cloudy_map = {f.name: f for f in cloudy_files}
        clear_map = {f.name: f for f in clear_files}

        common_names = sorted(set(cloudy_map.keys()) & set(clear_map.keys()))
        print(f"Found {len(common_names)} raw pairs before filtering.")

        self.pairs = []
        dropped = 0

        for name in common_names:
            c_path = cloudy_map[name]
            t_path = clear_map[name]

            c_img = np.array(Image.open(c_path).convert("RGB"))
            t_img = np.array(Image.open(t_path).convert("RGB"))

            c_zero_ratio = np.mean(c_img == 0)
            t_zero_ratio = np.mean(t_img == 0)

            if (c_zero_ratio >= self.zero_thr) or (t_zero_ratio >= self.zero_thr):
                dropped += 1
                continue

            self.pairs.append((c_path, t_path))

        print(f"Filtered pairs: {len(self.pairs)} (dropped {dropped} pairs)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cloudy_path, clear_path = self.pairs[idx]

        cloudy = Image.open(cloudy_path).convert("RGB")
        clear = Image.open(clear_path).convert("RGB")

        if cloudy.size != clear.size:
            w = min(cloudy.size[0], clear.size[0])
            h = min(cloudy.size[1], clear.size[1])
            cloudy = cloudy.crop((0, 0, w, h))
            clear = clear.crop((0, 0, w, h))

        if self.transform is not None:
            cloudy = self.transform(cloudy)
            clear = self.transform(clear)
        else:
            to_tensor = transforms.ToTensor()
            cloudy = to_tensor(cloudy)
            clear = to_tensor(clear)

        return cloudy, clear

# =============================================================================
# TRAINING
# =============================================================================
def train_unet(cloudy_patches, clear_patches, epochs=100, batch_size=16, lr=0.001, model_dir="unet_model"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = PatchDataset(cloudy_patches, clear_patches, zero_thr=0.8)
    if len(dataset) == 0:
        print("No paired patches found!")
        return
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    
    os.makedirs(model_dir, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for cloudy, clear in dataloader:
            cloudy, clear = cloudy.to(device), clear.to(device)
            
            optimizer.zero_grad()
            pred = model(cloudy)
            loss = criterion(pred, clear)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
        
        torch.save(model.state_dict(), f'{model_dir}/epoch_{epoch+1:03d}.pth')
    
    print(f"Training completed! Models saved to {model_dir}/")

# =============================================================================
# TESTING
# =============================================================================
def test_single_patch(model_path, test_cloudy_patch, output_path='result.png'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    img = Image.open(test_cloudy_patch).convert('RGB')
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(tensor)
    
    result = transforms.ToPILImage()(pred.squeeze().cpu().clamp(0,1))
    result.save(output_path)
    print(f"Saved single patch result: {output_path}")

def test_all_patches_and_merge(model_path, cloudy_patches_dir, output_path='merged_result.png'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    cloudy_dir = Path(cloudy_patches_dir)
    
    patches_info = []
    for cloudy_patch in cloudy_dir.glob("*.png"):
        name = cloudy_patch.stem
        if '_' in name and 'y' in name and 'x' in name:
            try:
                base_name = name.rsplit('_y', 1)[0]
                y_str, x_str = name.split('_y')[-1].split('_x')
                y, x = int(y_str), int(x_str)
                
                patches_info.append({
                    'cloudy_path': str(cloudy_patch),
                    'base_name': base_name,
                    'y': y * 256,
                    'x': x * 256
                })
            except:
                continue
    
    scenes = {}
    for info in patches_info:
        base_name = info['base_name']
        if base_name not in scenes:
            scenes[base_name] = {'patches': [], 'H': 0, 'W': 0}
        scenes[base_name]['patches'].append(info)
        scenes[base_name]['H'] = max(scenes[base_name]['H'], info['y'] + 256)
        scenes[base_name]['W'] = max(scenes[base_name]['W'], info['x'] + 256)
    
    all_results = []
    for scene_name, scene_data in scenes.items():
        print(f"Processing scene: {scene_name}")
        
        patches = sorted(scene_data['patches'], key=lambda p: (p['y'], p['x']))
        H, W = scene_data['H'], scene_data['W']
        
        result = torch.zeros((3, H, W), device=device)
        count = torch.zeros((H, W), device=device)
        
        with torch.no_grad():
            for patch_info in patches:
                cloudy_patch = Image.open(patch_info['cloudy_path']).convert('RGB')
                tensor = transforms.ToTensor()(cloudy_patch).unsqueeze(0).to(device)
                
                pred = model(tensor).squeeze(0)
                
                y, x = patch_info['y'], patch_info['x']
                result[:, y:y+256, x:x+256] += pred
                count[y:y+256, x:x+256] += 1
        
        result = result / count.unsqueeze(0).clamp(min=1)
        all_results.append((result.cpu(), scene_name))
    
    if len(all_results) == 1:
        result_img = transforms.ToPILImage()(all_results[0][0].clamp(0,1))
        result_img.save(output_path)
        print(f"Saved merged result: {output_path}")
    else:
        for i, (result_tensor, scene_name) in enumerate(all_results):
            result_img = transforms.ToPILImage()(result_tensor.clamp(0,1))
            out_path = output_path.replace('.png', f"_{scene_name[:20]}.png")
            result_img.save(out_path)
            print(f"Saved: {out_path}")

# =============================================================================
# MAIN CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Cloud Removal U-Net")
    parser.add_argument('--mode', required=True, choices=['preprocess', 'train', 'test', 'merge'],
                        help="Execution mode")
    parser.add_argument('--cloudy', help="Cloudy images/patches directory")
    parser.add_argument('--clear', help="Clear images/patches directory") 
    parser.add_argument('--model', help="Model path (.pth)")
    parser.add_argument('--test-patch', help="Single test patch path")
    parser.add_argument('--output', default='result.png', help="Output path")
    parser.add_argument('--epochs', type=int, default=100, help="Training epochs")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    
    args = parser.parse_args()
    
    print(f"=== Cloud Removal U-Net ({args.mode} mode) ===")
    
    if args.mode == 'preprocess':
        if not args.cloudy or not args.clear:
            print("Error: --cloudy and --clear required for preprocess")
            return
        os.makedirs("patches/cloudy_256", exist_ok=True)
        os.makedirs("patches/clear_256", exist_ok=True)
        
        print("Splitting cloudy images...")
        split_images_to_patches(args.cloudy, "patches/cloudy_256")
        print("Splitting clear images...")
        split_images_to_patches(args.clear, "patches/clear_256")
        
    elif args.mode == 'train':
        train_unet("patches/cloudy_256", "patches/clear_256", 
                  args.epochs, args.batch_size, args.lr)
    
    elif args.mode == 'test':
        if not args.model or not args.test_patch:
            print("Error: --model and --test-patch required")
            return
        test_single_patch(args.model, args.test_patch, args.output)
    
    elif args.mode == 'merge':
        if not args.model:
            print("Error: --model required")
            return
        test_all_patches_and_merge(args.model, "patches/cloudy_256", args.output)

if __name__ == "__main__":
    main()
