from dataset import cityscapes
from EfficientPS_modified import EfficientPS 
from torch.optim.lr_scheduler import _LRScheduler
import torch
from torch.utils.data import DataLoader
from torch import optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

from dataset import id2trainId, IMAGE_SIZE
from utils import train_one_epochs, val_one_epochs, predict_and_plot

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9):
        self.max_iters = max_iters
        self.power = power
        super().__init__(optimizer)
    
    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.max_iters) ** self.power for base_lr in self.base_lrs]
    
def configure_optimizer(model: EfficientPS, optim_algo='adam', backbone_lr=3e-5, default_lr=1e-3):
    backbone_params = [param for param in model.backbone.parameters() if param.requires_grad]
    fpn_params = [param for param in model.fpn.parameters() if param.requires_grad]
    semantic_params = [param for param in model.semantic_head.parameters() if param.requires_grad]
    param_groups = [
        {'params': backbone_params, 'lr': backbone_lr},  # 3e-5
        {'params': fpn_params, 'lr': default_lr},        # 1e-3
        {'params': semantic_params, 'lr': default_lr * 2}  # 2e-3
    ]
    if optim_algo == 'adam':
        optimizer = optim.Adam(param_groups)
    elif optim_algo == 'sgd':
        optimizer = optim.SGD(param_groups, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optim_algo}")
    return optimizer


if __name__ == '__main__': 
    BATCH_SIZE = 8
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_CLASSES = len(id2trainId)
    ROOT = '/kaggle/input/cityscapes-image-pairs/cityscapes_data'

    image_transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.1),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    ])

    train_transform = A.Compose([
        A.Resize(height=IMAGE_SIZE[1], width=IMAGE_SIZE[0], p=1.0),
        A.HorizontalFlip(p=0.5),  
        A.Rotate(limit=15, p=0.5),
        ToTensorV2(),
    ])  

    val_transform = A.Compose([
        A.Resize(height=IMAGE_SIZE[1], width=IMAGE_SIZE[0]),
        ToTensorV2(),
    ])

trainset = cityscapes(ROOT, subset='train', transform=train_transform, image_transform=image_transform)
valset = cityscapes(ROOT, subset='val', transform=val_transform, image_transform=None)


trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = EfficientPS(
    net_id=5,
    classes=N_CLASSES,
    image_size=(IMAGE_SIZE[1], IMAGE_SIZE[0])
).to(DEVICE)

best_iou = 0.0
EPOCHS = 20
optimizer = configure_optimizer(model, 'adam', 3e-5, 0.00363)
scheduler = PolyLR(optimizer, EPOCHS * len(trainloader))

train_loss_capture = [] 
val_loss_capture = [] 
val_iou_capture = []

for epoch in range(EPOCHS):
    train_loss = train_one_epochs(model, trainloader, optimizer, scheduler, DEVICE)
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}")

    val_loss, val_iou = val_one_epochs(model, valloader, DEVICE, N_CLASSES)
    print(f"Epoch {epoch+1}/{EPOCHS}, Val Loss: {val_loss:.4f}, Val mIoU: {val_iou:.2f}")

    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(), 'results/best_efficientps.pth')
        print(f"Saved best model with mIoU: {best_iou:.2f}")
        
    train_loss_capture.append(train_loss)
    val_loss_capture.append(val_loss)
    val_iou_capture.append(val_iou)

file_path = 'results/result.csv'
epoch_range = list(range(EPOCHS))

results = {'epoch': epoch_range, 'train_loss': train_loss_capture, 'val_loss': val_loss_capture, 'val_iou': val_iou_capture}

df = pd.DataFrame(results)
df.to_csv(file_path, index=False)

print("\nTesting the best model...")
model.load_state_dict(torch.load('results/best_efficientps.pth'))
test_loss, test_iou = val_one_epochs(model, valloader, DEVICE, N_CLASSES)
print(f"Test Loss: {test_loss:.4f}, Test mIoU: {test_iou:.2f}")

idx = torch.randint(0, len(valset), (1, )).item()

predict_and_plot(model, valset, idx, DEVICE)
