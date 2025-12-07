import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split
from torch import optim
from tqdm import tqdm
import numpy as np

from dataset import SegmentationDataset
from unet import UNet
from metrics import BCEDiceLoss, dice_coefficient, iou_score, pixel_accuracy
from utils import plot_history

def train_model(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create dataset
    full_dataset = SegmentationDataset(args.img_dir, args.mask_dir)
    
    # Split into train and val
    n_val = int(len(full_dataset) * 0.2)
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val])
    
    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f'Training with {n_train} images, validating with {n_val} images.')

    # Initialize model
    model = UNet(n_channels=3, n_classes=1).to(device)
    
    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = BCEDiceLoss()
    
    # Training loop
    best_dice = 0.0
    train_losses = []
    val_losses = []
    val_dices = []
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        with tqdm(total=n_train, desc=f'Epoch {epoch+1}/{args.epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, masks = batch
                images = images.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                masks_pred = model(images)
                loss = criterion(masks_pred, masks)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss': loss.item()})
                pbar.update(images.shape[0])
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        dice_score = 0
        iou = 0
        acc = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images, masks = batch
                images = images.to(device)
                masks = masks.to(device)
                
                masks_pred = model(images)
                loss = criterion(masks_pred, masks)
                val_loss += loss.item()
                
                dice_score += dice_coefficient(masks_pred, masks).item()
                iou += iou_score(masks_pred, masks).item()
                acc += pixel_accuracy(masks_pred, masks).item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_dice = dice_score / len(val_loader)
        avg_iou = iou / len(val_loader)
        avg_acc = acc / len(val_loader)
        
        val_losses.append(avg_val_loss)
        val_dices.append(avg_dice)
        
        print(f'Val Loss: {avg_val_loss:.4f} | Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f} | Acc: {avg_acc:.4f}')
        
        # Save best model
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print('Model saved!')
            
    # Plot history
    plot_history(train_losses, val_losses, val_dices, save_path=os.path.join(args.checkpoint_dir, 'training_plot.png'))
    print('Training finished!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--img_dir', type=str, default='data/images', help='Directory for images')
    parser.add_argument('--mask_dir', type=str, default='data/masks', help='Directory for masks')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    
    args = parser.parse_args()
    train_model(args)
