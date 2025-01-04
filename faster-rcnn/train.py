import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from PIL import Image
import time
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import json
import xml.etree.ElementTree as ET
import warnings
import matplotlib.pyplot as plt

# CONFIG
batch_size = 16
learning_rate = 0.005
num_epochs = 300  
num_workers = 8
threshold = 0.7
optimizer_type = 'SGD'
# optimizer_type = 'Adam'
momentum = 0.9
weight_decay = 0.0005
patience = 50
best_val_loss = float('inf')
epochs_without_improvement = 0

#input paths
train_dir = 'data\\train'
valid_dir = 'data\\valid'
test_dir = 'data\\test'

#output path
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def get_train_transform():
    return A.Compose([
        A.Normalize(),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_valid_transform():
    return A.Compose([
        A.Normalize(),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

class CustomDataset(Dataset):
    def __init__(self, root, transforms=None, class_name_to_id=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "annotations"))))
        if class_name_to_id is None:
            self.class_name_to_id = {'couple': 1} 
        else:
            self.class_name_to_id = class_name_to_id

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        ann_path = os.path.join(self.root, "annotations", self.annotations[idx])
        
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        
        boxes = []
        labels = []
        
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        for obj in root.findall('object'):
            label = obj.find('name').text
            label_id = self.class_name_to_id.get(label, None)
            if label_id is None:
                continue
            
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_id)
        
        if len(boxes) > 0:
            boxes = np.array(boxes)
            labels = np.array(labels)
        else:
            boxes = np.empty((0, 4), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int64)
        
        if self.transforms is not None:
            transformed = self.transforms(image=img, bboxes=boxes, labels=labels)
            img = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    train_dataset = CustomDataset(train_dir, transforms=get_train_transform())
    valid_dataset = CustomDataset(valid_dir, transforms=get_valid_transform())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    
    num_classes = 2  #background + class 'couple'
    model = get_model(num_classes)
    
    params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    metric = MeanAveragePrecision().to(device)
    train_loss_list = []
    val_loss_list = []
    map_list = []
    map50_list = []
    map50_95_list = []
    
    plt.switch_backend('Agg')
    
    #training
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        start_time = time.time()
        
        model.train()
        train_loss = 0
        train_classifier_loss = 0
        train_box_loss = 0
        train_objectness_loss = 0
        train_rpn_box_loss = 0
    
        for images, targets in tqdm(train_loader, desc="Trening"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            train_loss += losses.item()
            train_classifier_loss += loss_dict['loss_classifier'].item()
            train_box_loss += loss_dict['loss_box_reg'].item()
            train_objectness_loss += loss_dict['loss_objectness'].item()
            train_rpn_box_loss += loss_dict['loss_rpn_box_reg'].item()
        
        train_loss /= len(train_loader)
        train_classifier_loss /= len(train_loader)
        train_box_loss /= len(train_loader)
        train_objectness_loss /= len(train_loader)
        train_rpn_box_loss /= len(train_loader)
        
        #validation
        val_loss = 0
        val_classifier_loss = 0
        val_box_loss = 0
        val_objectness_loss = 0
        val_rpn_box_loss = 0
        metric.reset()
        
        with torch.no_grad():
            for images, targets in tqdm(valid_loader, desc="validation"):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                model.train() #just for loss calculating purposes, weights are not updated
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                val_loss += losses.item()
                val_classifier_loss += loss_dict['loss_classifier'].item()
                val_box_loss += loss_dict['loss_box_reg'].item()
                val_objectness_loss += loss_dict['loss_objectness'].item()
                val_rpn_box_loss += loss_dict['loss_rpn_box_reg'].item()
                model.eval()
            
                outputs = model(images)
                metric.update(outputs, targets)
            
            val_loss /= len(valid_loader)
            val_classifier_loss /= len(valid_loader)
            val_box_loss /= len(valid_loader)
            val_objectness_loss /= len(valid_loader)
            val_rpn_box_loss /= len(valid_loader)
            map_metric = metric.compute()
            mAP = map_metric['map'].item()
            mAP50 = map_metric['map_50'].item()
            mAP50_95 = map_metric['map'].item()
            map_list.append(mAP)
            map50_list.append(mAP50)
            map50_95_list.append(mAP50_95)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        print(f"epoch time: {epoch_time:.2f}s, epoch number: {epoch+1}, mAP: {mAP:.4f}, mAP50: {mAP50:.4f}, mAP50-95: {mAP50_95:.4f}")
        print(f"total loss: {train_loss:.4f}, class. loss: {train_classifier_loss:.4f}, box reg. loss: {train_box_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.7f}")
        print("train losses :")
        print(f"classifier loss : {train_classifier_loss}")
        print(f"box loss : {train_box_loss}")
        print(f"Objectiveness loss : {train_objectness_loss}")
        print(f"RPN box loss : {train_rpn_box_loss}")
        print(f"total loss : {train_loss}")
        print("validation losses :")
        print(f"Classifier loss : {val_classifier_loss}")
        print(f"box loss : {val_box_loss}")
        print(f"Objectiveness loss : {val_objectness_loss}")
        print(f"RPN box loss : {val_rpn_box_loss}")
        print(f"total loss : {val_loss}")
        print(f"mAP: {mAP:.4f}, mAP50: {mAP50:.4f}, mAP50-95: {mAP50_95:.4f}")
        
        train_loss_list.append({
            'epoch': epoch+1,
            'total_loss': train_loss,
            'classifier_loss': train_classifier_loss,
            'box_loss': train_box_loss,
            'objectness_loss': train_objectness_loss,
            'rpn_box_loss': train_rpn_box_loss,
        })
        val_loss_list.append({
            'epoch': epoch+1,
            'total_loss': val_loss,
            'classifier_loss': val_classifier_loss,
            'box_loss': val_box_loss,
            'objectness_loss': val_objectness_loss,
            'rpn_box_loss': val_rpn_box_loss,
            'mAP': mAP,
            'mAP50': mAP50,
            'mAP50-95': mAP50_95,
        })
        
        #check if should early stop
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            #save best model
            torch.save(model.state_dict(), os.path.join(output_dir, f"best_model.pth"))
            print(f"new best validation loss: {best_val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"no improvement in {epochs_without_improvement} epochs")
            if epochs_without_improvement >= patience:
                print(f"Early stopping the training")
                break 

        #model save
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch+1}.pth"))
        
        #losses plot
        epochs = [x['epoch'] for x in train_loss_list]
        train_total_losses = [x['total_loss'] for x in train_loss_list]
        val_total_losses = [x['total_loss'] for x in val_loss_list]
        
        plt.figure(figsize=(10,5))
        plt.plot(epochs, train_total_losses, label='Train total loss')
        plt.plot(epochs, val_total_losses, label='validation total loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
        plt.close()
        
        #map plot
        plt.figure(figsize=(10,5))
        plt.plot(epochs, map_list, label='mAP')
        plt.plot(epochs, map50_list, label='mAP50')
        plt.plot(epochs, map50_95_list, label='mAP50-95')
        plt.xlabel('epoch')
        plt.ylabel('mAP')
        plt.title('mAP')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'map_plot.png'))
        plt.close()
        
    with open(os.path.join(output_dir, 'train_losses.json'), 'w') as f:
        json.dump(train_loss_list, f)
    
    with open(os.path.join(output_dir, 'val_losses.json'), 'w') as f:
        json.dump(val_loss_list, f)
