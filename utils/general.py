import os 
import numpy as np
import torch 
import matplotlib.pyplot as plt
from PIL import Image
from dataset import id2trainId, id2color 

EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


def decode_segmap_torch(enc, id2color, id2trainId):
    H, W = enc.shape
    device = enc.device

    num_classes = 19 
    color_map = torch.zeros((num_classes + 1, 3), dtype=torch.uint8, device=device)  
    trainId_to_index = {} 

    for id_, trainId in id2trainId.items():
        color_map[trainId] = torch.tensor(id2color[id_], dtype=torch.uint8, device=device)
        trainId_to_index[trainId] = trainId

    color_map[num_classes] = torch.tensor([0, 0, 0], dtype=torch.uint8, device=device) 

    output = torch.full_like(enc, num_classes, dtype=torch.long, device=device)  
    for trainId, index in trainId_to_index.items():
        output[enc == trainId] = index

    rgb = color_map[output]  # [H, W, 3]

    return rgb

def visualize_sample(index, dataset): 
    img, label = dataset[index]
    img = img.numpy() 
    img = np.transpose(img, (1, 2, 0))
    label = decode_segmap_torch(label, id2color, id2trainId)
    label = label.numpy()
    
    plt.figure(figsize=(10, 10))

    plt.subplot(1, 2, 1)
    plt.title('Image') 
    plt.axis('off')

    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.title('Mask')
    plt.imshow(label)

    plt.axis('off')
    plt.show()

def predict_and_plot(model, valset, index, device):
    img, label = valset[index]
    img = img.unsqueeze(0)
    label = label.unsqueeze(0)
    
    img = img.to(device)
    label = label.to(device) 
    
    model.eval()
    with torch.no_grad():
        out, _ = model(img, label)
    
    out = out.squeeze(0)
    label = label.squeeze(0).cpu()
    pred = torch.argmax(out, dim=0)
    pred_rgb = decode_segmap_torch(pred.cpu(), id2color, id2trainId)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Hình ảnh gốc')
    plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Nhãn')
    plt.imshow(decode_segmap_torch(label, id2color, id2trainId))
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Dự đoán')
    plt.imshow(pred_rgb)
    plt.axis('off')

    plt.show()