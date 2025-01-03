import os
import yaml
import argparse
import time
import torch
from ultralytics import YOLO
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO model training")
    parser.add_argument('--config', type=str, default='config.yml')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    args = parse_args()
    config = load_config(args.config)

    #CONFIG
    batch_size = config.get('batch_size', 8)
    learning_rate = config.get('learning_rate', 1e-3)
    num_epochs = config.get('num_epochs', 2)
    num_workers = config.get('num_workers', 8)
    optimizer_type = config.get('optimizer', 'sgd')
    dataset_path = config.get('dataset_path', 'dataset')
    output_dir = config.get('output_dir', 'outputs')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    imgsz = config.get('imgsz', 512)

    os.makedirs(output_dir, exist_ok=True)

    #MODEL YOLO
    model = YOLO('yolo11n.pt')
    model.to(device)

    train_path = os.path.join(dataset_path, 'train')
    valid_path = os.path.join(dataset_path, 'valid')
    data = {
        'train': os.path.abspath(os.path.join(train_path, 'images')),
        'val': os.path.abspath(os.path.join(valid_path, 'images')),
        'test': os.path.abspath(os.path.join(dataset_path, 'test', 'images'))}
    data_yaml = os.path.join(output_dir, 'data.yaml')
    with open(data_yaml, 'w') as f:
        yaml.dump({
            'train': data['train'],
            'val': data['val'],
            'test': data['test'],
            'nc': config.get('num_classes', 1),
            'names': config.get('class_names', ['couple'])
        }, f)

    #HYPERPARAMETERS
    training_params = {
        'data': data_yaml,
        'epochs': num_epochs,
        'batch': batch_size,
        'lr0': learning_rate,
        'workers': num_workers,
        'device': device,
        'project': output_dir,
        'name': 'yolo_training',
        'exist_ok': True,
        'optimizer': optimizer_type,
        'imgsz': imgsz
    }

    #TRAINING
    start_time = time.time()
    results = model.train(**training_params, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, scale=0.0, fliplr=0.0, mosaic=0.0, patience=50, erasing=0.0, plots=True)
    total_time = time.time() - start_time
    print(f"Total training time: {total_time/3600:.2f} hours")

    #SAVING MODEL
    best_model_path = os.path.join(output_dir, 'best_model.pt')
    last_model_path = os.path.join(output_dir, 'last_model.pt')
    model.save(best_model_path)
    model.save(last_model_path)
    print(f"model saved as: {best_model_path} and {last_model_path}")

    #log
    log_path = os.path.join(output_dir, 'training_log.txt')
    with open(log_path, 'w') as f:
        f.write(str(results))
    print(f"logs: {log_path}")

    #LOSSES
    try:
        log_csv = os.path.join(output_dir, 'yolo_training', 'results.csv')
        if os.path.exists(log_csv):
            df = pd.read_csv(log_csv)
            plt.figure(figsize=(10, 5))
            plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss')
            plt.plot(df['epoch'], df['train/cls_loss'], label='Class Loss')
            plt.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss')
            plt.plot(df['epoch'], df['train/total_loss'], label='Total Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('training losses')
            plt.legend()
            loss_plot_path = os.path.join(output_dir, 'loss_plot.png')
            plt.savefig(loss_plot_path)
            plt.close()
            print(f"loss plot {loss_plot_path}")
        else:
            print(f"cant find log csv {log_csv}")

if __name__ == "__main__":
    main()
