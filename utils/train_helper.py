import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from tqdm import tqdm

def compute_iou(preds, masks, num_classes):
    preds = preds.view(-1).cpu().numpy()
    labels = masks.view(-1).cpu().numpy()
    valid_mask = labels != 250
    preds = preds[valid_mask]
    labels = labels[valid_mask]
    
    cm = confusion_matrix(labels, preds, labels=range(num_classes))
    ious = []
    for i in range(num_classes):
        intersection = cm[i, i]
        union = cm[i, :].sum() + cm[:, i].sum() - intersection
        ious.append(intersection / union if union > 0 else float('nan'))
    return np.nanmean(ious)

def compute_ap(outputs, masks, num_classes, ignore_index=250):
    probs = torch.softmax(outputs, dim=1)
    probs = probs.permute(0, 2, 3, 1).reshape(-1, num_classes)
    masks = masks.view(-1)

    valid_mask = masks != ignore_index
    probs = probs[valid_mask]
    masks = masks[valid_mask]

    aps = []
    for cls in range(num_classes):
        cls_probs = probs[:, cls].cpu().numpy()
        cls_labels = (masks == cls).cpu().numpy().astype(np.int32)

        if cls_labels.sum() == 0:
            continue

        ap = average_precision_score(cls_labels, cls_probs)
        aps.append(ap)

    return np.mean(aps) if aps else float('nan')
    
def train_one_epochs(model, dataloader, optimizer, scheduler=None, device='cpu'):
    model.train()
    epoch_loss = 0
  
    for img, masks in tqdm(dataloader, desc='training'):
        img, masks = img.to(device), masks.to(device).long()

        optimizer.zero_grad()
        outputs, loss = model(img, targets=masks)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()


        if scheduler != None: 
            scheduler.step()
        
    return epoch_loss / len(dataloader)

def val_one_epochs(model, dataloader, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    num_batches = 0
    with torch.inference_mode():
        for img, masks in tqdm(dataloader, desc='evaluating'):
            img, masks = img.to(device), masks.to(device).long()
            outputs, loss = model(img, masks)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            iou = compute_iou(preds.cpu(), masks.cpu(), num_classes)
            total_iou += iou
            num_batches += 1
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    return avg_loss, avg_iou