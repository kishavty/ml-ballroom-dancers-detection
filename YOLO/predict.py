import os
import torch
from ultralytics import YOLO
import yaml
import time
import pandas as pd

confidence_threshold = 0.7
image_size = 512

base_dir = "outputs\\yolo5"
test_images = "dataset\\test\\images"
test_labels = "dataset\\test\\labels"
output_dir = "outputs\\predictions"
data_yaml_path = "temp_data.yaml"

os.makedirs(output_dir, exist_ok=True)

def create_data_yaml(test_images_path, test_labels_path, num_classes, class_names):
    train_dummy_path = test_images_path
    val_dummy_path = test_images_path

    data = {
        'train': os.path.abspath(train_dummy_path),  #dummy
        'val': os.path.abspath(val_dummy_path),      #dummy
        'test': os.path.abspath(test_images_path),
        'nc': num_classes,
        'names': class_names,
    }
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data, f)

def run_model(model_path, model_name):
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO(model_path)
        model.to(device)
        num_classes = model.model.nc
        class_names = model.model.names
        create_data_yaml(test_images, test_labels, num_classes, class_names)

        print(f"Evaluating model: {model_name}")
        start_time = time.time()
        results = model.val(
            data=data_yaml_path,
            task='detect',
            conf=confidence_threshold,
            imgsz=image_size
        )
        end_time = time.time()
        try:
            mAP50 = results.maps.get('AP50', None)
            mAP50_95 = results.maps.get('AP50-95', None)
            time_per_image = results.speed.get('inference', None)
            total_time = end_time - start_time

            print("Results:")
            print(f"mAP50: {mAP50}")
            print(f"mAP50-95: {mAP50_95}")
            print(f"time per image (ms): {time_per_image}")
            print(f"total time (s): {total_time}")
        except:
            print("Warning: could not parse metrics from results object")

        preds = model.predict(
            source=test_images, 
            conf=confidence_threshold,
            imgsz=image_size,
            save=True,
            project=output_dir,
            name=model_name
        )
        print(f"Annotated images saved in: {os.path.join(output_dir, model_name)}")

    except Exception as e:
        print(f"Error evaluating model {model_name}: {e}")

    finally:
        if os.path.exists(data_yaml_path):
            os.remove(data_yaml_path)

for batch in ["batch4", "batch8", "batch16"]:
    for lr in ["lr0-001", "lr0-005"]:
        for optimizer in ["adam", "sgd"]:
            model_name = f"{batch}_{lr}_{optimizer}"
            model_path = os.path.join(base_dir, batch, lr, optimizer, "best_model.pt")
            if os.path.exists(model_path):
                run_model(model_path, model_name)
            else:
                print(f"model not found: {model_path}")
