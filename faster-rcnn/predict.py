import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import time
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

#CONFIG
num_classes = 2  #background+ class 'couple'
model_path = 'outputs/batch16/lr0-005/sgd/model_epoch_7.pth'
test_dir = 'test'
annotations_xml_dir = 'data/test/annotations'
images_dir = os.path.join("data", test_dir, "images")
threshold = 0.7
iou_threshold = 0.3
coco_annotations_file = 'data/test/annotations_coco.json'
output_inference_dir = 'pics-inference'

os.makedirs(output_inference_dir, exist_ok=True)

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_test_transform():
    return A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])

def parse_voc_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    image_info = {}
    objects = []
    for elem in root:
        if elem.tag == 'filename':
            image_info['filename'] = elem.text
        elif elem.tag == 'size':
            image_info['width'] = int(elem.find('width').text)
            image_info['height'] = int(elem.find('height').text)
        elif elem.tag == 'object':
            obj = {}
            obj['name'] = elem.find('name').text
            bndbox = elem.find('bndbox')
            obj['bbox'] = [
                int(bndbox.find('xmin').text),
                int(bndbox.find('ymin').text),
                int(bndbox.find('xmax').text),
                int(bndbox.find('ymax').text)
            ]
            objects.append(obj)
    image_info['objects'] = objects
    return image_info

def convert_voc_to_coco(annotations_xml_dir, images_dir, output_json):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "couple"}]
    }
    annotation_id = 1
    image_id_map = {}
    image_id = 1
    xml_files = [f for f in os.listdir(annotations_xml_dir) if f.endswith('.xml')]
    
    for xml_file in tqdm(xml_files):
        parsed = parse_voc_xml(os.path.join(annotations_xml_dir, xml_file))
        filename = parsed['filename']
        width = parsed['width']
        height = parsed['height']
        image_id_map[filename] = image_id
        coco['images'].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height})
        for obj in parsed['objects']:
            xmin, ymin, xmax, ymax = obj['bbox']
            width_box = xmax - xmin
            height_box = ymax - ymin
            coco['annotations'].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  #'couple'
                "bbox": [xmin, ymin, width_box, height_box],
                "area": width_box * height_box,
                "iscrowd": 0})
            annotation_id += 1
        image_id += 1
    with open(output_json, 'w') as f:
        json.dump(coco, f)
    return image_id_map

image_id_map = convert_voc_to_coco(annotations_xml_dir, images_dir, coco_annotations_file)
model = get_model(num_classes)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

#test data
test_images = list(sorted(os.listdir(images_dir)))
class_id_to_name = {1: 'couple'}

coco_gt = COCO(coco_annotations_file)
file_name_to_id = {img['file_name']: img['id'] for img in coco_gt.imgs.values()}
coco_results = []

single_image = test_images[0]
image_id_single = file_name_to_id.get(single_image)
if image_id_single is None:
    print(f"{single_image} image not found.")
else:
    img_path = os.path.join(images_dir, single_image)
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    transform = get_test_transform()
    transformed = transform(image=img_np)
    img_tensor = transformed['image'].unsqueeze(0).to(device)

    start_time_single = time.time()
    with torch.no_grad():
        output_single = model(img_tensor)
    end_time_single = time.time()
    inference_time_single = end_time_single - start_time_single
    print(f"inference on ({single_image}) took {inference_time_single:.4f} s")
    boxes = output_single[0]['boxes'].cpu().numpy()
    scores = output_single[0]['scores'].cpu().numpy()
    labels = output_single[0]['labels'].cpu().numpy()

    #nms
    keep_indices = nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold).numpy()
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    labels = labels[keep_indices]
    indices = np.where(scores > threshold)[0]
    boxes = boxes[indices]
    scores = scores[indices]
    labels = labels[indices]
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_np)

    for box, score, label in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, 
            linewidth=2, edgecolor='blue', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            xmin, ymin - 10, 
            f"{class_id_to_name[label]}: {score:.2f}", 
            color='white', fontsize=10, 
            bbox=dict(facecolor='blue', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
        )
    
    plt.axis('off')
    output_image_path = os.path.join(output_inference_dir, single_image)
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"image saved in {output_image_path}")
start_time_total = time.time()

for idx, img_name in enumerate(tqdm(test_images, desc="test data inference")):
    image_id = file_name_to_id.get(img_name)
    if image_id is None:
        print(f"{img_name} image not found")
        continue

    img_path = os.path.join(images_dir, img_name)
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    transform = get_test_transform()
    transformed = transform(image=img_np)
    img_tensor = transformed['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    boxes = outputs[0]['boxes']
    scores = outputs[0]['scores']
    labels = outputs[0]['labels']

    keep_indices = nms(boxes, scores, iou_threshold).numpy()
    boxes = boxes[keep_indices].numpy()
    scores = scores[keep_indices].numpy()
    labels = labels[keep_indices].numpy()
    indices = np.where(scores > threshold)[0]
    boxes = boxes[indices]
    scores = scores[indices]
    labels = labels[indices]
    for box, score, label in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        coco_result = {
            "image_id": image_id,
            "category_id": int(label),
            "bbox": [float(xmin), float(ymin), float(width), float(height)],
            "score": float(score)}
        coco_results.append(coco_result)

    if len(boxes) > 0:
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img_np)

        for box, score, label in zip(boxes, scores, labels):
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, 
                linewidth=2, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)
            ax.text(
                xmin, ymin - 10, 
                f"{class_id_to_name[label]}: {score:.2f}", 
                color='white', fontsize=10, 
                bbox=dict(facecolor='blue', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        plt.axis('off')
        output_image_path = os.path.join(output_inference_dir, img_name)
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

end_time_total = time.time()
inference_time_total = end_time_total - start_time_total
average_time_per_image = inference_time_total / len(test_images)
print(f"inference time on all test data: {inference_time_total:.4f} s.")
print(f"average inference time on an image: {average_time_per_image:.4f} s")

results_file = 'coco_results.json'
with open(results_file, 'w') as f:
    json.dump(coco_results, f)

coco_dt = coco_gt.loadRes(results_file)
coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()