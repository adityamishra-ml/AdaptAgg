# client.py - COMPLETE FIXED VERSION
import argparse
import json
import os
import shutil
from pathlib import Path
import yaml
import numpy as np
from collections import defaultdict
import random
import warnings

import flwr as fl
import torch
from ultralytics import YOLO

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================
FIXED_CLASS_ORDER = ['bike', 'bus', 'car', 'motor', 'person', 'rider', 'truck']
MAX_TRAIN_SAMPLES_PER_CLASS = 15000
MAX_VAL_SAMPLES_PER_CLASS = 2000

# ============================================================================
# DATA PRE-PROCESSING (SAME AS BEFORE)
# ============================================================================

class BalancedDatasetCreator:
    def __init__(self, coco_json_path, images_dir, target_samples_per_class):
        self.coco_json_path = coco_json_path
        self.images_dir = images_dir
        self.target_samples_per_class = target_samples_per_class
        with open(coco_json_path, 'r') as f:
            self.coco_data = json.load(f)
        self.create_mappings()

    def create_mappings(self):
        self.image_id_to_filename = {img['id']: img['file_name'] for img in self.coco_data['images']}
        self.image_id_to_size = {img['id']: (img['width'], img['height']) for img in self.coco_data['images']}
        coco_category_name_to_id = {cat['name']: cat['id'] for cat in self.coco_data['categories']}
        self.category_to_class = {}
        self.class_names = []
        for class_idx, class_name in enumerate(FIXED_CLASS_ORDER):
            if class_name in coco_category_name_to_id:
                coco_cat_id = coco_category_name_to_id[class_name]
                self.category_to_class[coco_cat_id] = class_idx
                self.class_names.append(class_name)

    def analyze_class_distribution(self):
        class_to_images = defaultdict(set)
        for ann in self.coco_data['annotations']:
            if ann['category_id'] in self.category_to_class:
                class_idx = self.category_to_class[ann['category_id']]
                class_to_images[class_idx].add(ann['image_id'])
        return class_to_images

    def create_balanced_subset(self):
        class_to_images = self.analyze_class_distribution()
        selected_images = set()
        for class_idx in range(len(self.class_names)):
            available_images = list(class_to_images.get(class_idx, set()))
            if not available_images: continue
            target = self.target_samples_per_class
            if len(available_images) <= target:
                selected_images.update(available_images)
            else:
                selected_images.update(random.sample(available_images, target))
        return selected_images

    def filter_coco_data(self, selected_image_ids):
        filtered_data = {
            'info': self.coco_data.get('info', {}),
            'licenses': self.coco_data.get('licenses', []),
            'categories': [{'id': i, 'name': name} for i, name in enumerate(self.class_names)],
            'images': [],
            'annotations': []
        }
        old_to_new_image_id, new_image_id = {}, 0
        for img in self.coco_data['images']:
            if img['id'] in selected_image_ids:
                old_to_new_image_id[img['id']] = new_image_id
                img_copy = img.copy()
                img_copy['id'] = new_image_id
                filtered_data['images'].append(img_copy)
                new_image_id += 1
        
        new_ann_id = 0
        for ann in self.coco_data['annotations']:
            if ann['image_id'] in selected_image_ids and ann['category_id'] in self.category_to_class:
                ann_copy = ann.copy()
                ann_copy['id'] = new_ann_id
                ann_copy['image_id'] = old_to_new_image_id[ann['image_id']]
                ann_copy['category_id'] = self.category_to_class[ann['category_id']]
                filtered_data['annotations'].append(ann_copy)
                new_ann_id += 1
        return filtered_data

class ImprovedCOCOToYOLOConverter:
    def __init__(self, coco_data, images_dir, output_dir):
        self.coco_data = coco_data
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        self.image_id_to_size = {img['id']: (img['width'], img['height']) for img in coco_data['images']}
        self.class_names = [cat['name'] for cat in sorted(coco_data['categories'], key=lambda x: x['id'])]

    def setup_directories(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)
        (self.output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels').mkdir(parents=True, exist_ok=True)

    def convert(self):
        self.setup_directories()
        image_annotations = defaultdict(list)
        for ann in self.coco_data['annotations']:
            image_annotations[ann['image_id']].append(ann)

        for img_id, annotations in image_annotations.items():
            filename = self.image_id_to_filename[img_id]
            img_width, img_height = self.image_id_to_size[img_id]
            label_path = self.output_dir / 'labels' / f"{Path(filename).stem}.txt"
            
            with open(label_path, 'w') as f:
                for ann in annotations:
                    x, y, w, h = ann['bbox']
                    cx = (x + w / 2) / img_width
                    cy = (y + h / 2) / img_height
                    nw = w / img_width
                    nh = h / img_height
                    f.write(f"{ann['category_id']} {cx} {cy} {nw} {nh}\n")
            
            src_path = self.images_dir / filename
            dst_path = self.output_dir / 'images' / filename
            if src_path.exists():
                shutil.copy2(src_path, dst_path)

def prepare_client_dataset(cid: int):
    client_dir = Path(f"clients/client_{cid}")
    print(f"CLIENT {cid}: 📊 Preparing dataset from {client_dir}")

    train_balancer = BalancedDatasetCreator(
        client_dir / 'train.json', client_dir / 'train_images', MAX_TRAIN_SAMPLES_PER_CLASS
    )
    selected_train_ids = train_balancer.create_balanced_subset()
    balanced_train_data = train_balancer.filter_coco_data(selected_train_ids)
    train_converter = ImprovedCOCOToYOLOConverter(
        balanced_train_data, client_dir / 'train_images', client_dir / 'yolo_train'
    )
    train_converter.convert()

    val_balancer = BalancedDatasetCreator(
        client_dir / 'val.json', client_dir / 'val_images', MAX_VAL_SAMPLES_PER_CLASS
    )
    selected_val_ids = val_balancer.create_balanced_subset()
    balanced_val_data = val_balancer.filter_coco_data(selected_val_ids)
    val_converter = ImprovedCOCOToYOLOConverter(
        balanced_val_data, client_dir / 'val_images', client_dir / 'yolo_val'
    )
    val_converter.convert()

    yolo_config = {
        'path': str(client_dir.absolute()),
        'train': 'yolo_train/images',
        'val': 'yolo_val/images',
        'nc': len(FIXED_CLASS_ORDER),
        'names': FIXED_CLASS_ORDER
    }
    yaml_path = client_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yolo_config, f, default_flow_style=False)
    
    print(f"CLIENT {cid}: ✅ Dataset ready. Config: {yaml_path}")
    return yaml_path, len(selected_val_ids)

# ============================================================================
# FIXED FLOWER CLIENT - FORCE 7 CLASSES
# ============================================================================

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, yaml_path, num_val_examples):
        self.cid = cid
        self.yaml_path = yaml_path
        self.num_val_examples = num_val_examples
        
        print(f"CLIENT {cid}: 🔧 Initializing model with {len(FIXED_CLASS_ORDER)} classes...")
        
        # === CRITICAL FIX: Initialize with correct number of classes ===
        # Step 1: Train once on data to force correct architecture
        temp_model = YOLO("yolov8s.pt")
        print(f"CLIENT {cid}: 🎯 Running initial training to set architecture...")
        temp_model.train(
            data=str(yaml_path),
            epochs=1,  # Just 1 epoch to set architecture
            imgsz=640,
            batch=4,
            verbose=False,
            workers=0,
            device=0,
            exist_ok=True,
            project=f'clients/client_{cid}/init_train',
            name='architecture_init'
        )
        
        # Step 2: Save this correctly-configured model as reference
        self.reference_path = Path(f"clients/client_{cid}/reference_model.pt")
        temp_model.save(self.reference_path)
        print(f"CLIENT {cid}: 💾 Reference model saved with {len(FIXED_CLASS_ORDER)} classes")
        
        # Step 3: Load it back as the main model
        self.model = YOLO(str(self.reference_path))
        
        # Verify architecture
        self.reference_state_keys = list(self.model.model.state_dict().keys())
        self.num_params = len(self.reference_state_keys)
        
        # Create save directory
        self.weights_dir = Path(f"clients/client_{cid}/saved_weights")
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"CLIENT {cid}: ✅ Model ready ({self.num_params} parameters, {len(FIXED_CLASS_ORDER)} classes)")

    def _reset_model(self):
        """Reset model to 7-class architecture"""
        self.model = YOLO(str(self.reference_path))

    def get_parameters(self, config):
        """Safely extract parameters"""
        parameters = []
        with torch.no_grad():
            for key in self.reference_state_keys:
                param = self.model.model.state_dict()[key]
                parameters.append(param.detach().cpu().numpy().copy())
        return parameters

    def set_parameters(self, parameters):
        """Safely load parameters"""
        if len(parameters) != self.num_params:
            raise ValueError(
                f"CLIENT {self.cid}: Parameter mismatch! "
                f"Expected {self.num_params}, got {len(parameters)}"
            )
        
        try:
            new_state_dict = {}
            for key, param_array in zip(self.reference_state_keys, parameters):
                current_param = self.model.model.state_dict()[key]
                new_tensor = torch.from_numpy(param_array).clone()
                new_tensor = new_tensor.to(device=current_param.device, dtype=current_param.dtype)
                new_state_dict[key] = new_tensor
            
            self.model.model.load_state_dict(new_state_dict, strict=True)
            
        except Exception as e:
            print(f"\nCLIENT {self.cid}: ❌ ERROR loading parameters!")
            print(f"Details: {str(e)[:500]}")
            raise

    def fit(self, parameters, config):
        """Train model"""
        current_round = config.get("round", 0)
        print(f"\n{'='*70}")
        print(f"CLIENT {self.cid}: 🏋️  Round {current_round}")
        print(f"{'='*70}")
        
        # Reset and load parameters
        print(f"CLIENT {self.cid}: 🔄 Resetting to reference model...")
        self._reset_model()
        
        print(f"CLIENT {self.cid}: 📥 Loading global parameters...")
        self.set_parameters(parameters)
        
        # Save before training
        before_path = self.weights_dir / f"round_{current_round}_before.pt"
        with torch.no_grad():
            torch.save(self.model.model.state_dict(), before_path)
        print(f"CLIENT {self.cid}: 💾 Saved pre-training weights")
        
        # Train
        train_dir = Path(self.yaml_path).parent / 'yolo_train' / 'images'
        num_train_examples = len(list(train_dir.glob('*')))
        print(f"CLIENT {self.cid}: 🎯 Training on {num_train_examples} images...")

        self.model.train(
            data=str(self.yaml_path),
            epochs=5,
            imgsz=640,
            batch=4,
            optimizer='SGD',
            lr0=0.001,
            workers=0,
            device=0,
            project=f'clients/client_{self.cid}/training_run',
            name=f'round_{current_round}',
            exist_ok=True,
            verbose=False
        )
        
        # Extract parameters immediately
        print(f"CLIENT {self.cid}: 📤 Extracting parameters...")
        trained_parameters = self.get_parameters(config={})
        
        # Save after training
        after_path = self.weights_dir / f"round_{current_round}_after.pt"
        with torch.no_grad():
            torch.save(self.model.model.state_dict(), after_path)
        print(f"CLIENT {self.cid}: 💾 Saved post-training weights")
        
        # Save final model
        total_rounds = config.get("num_rounds", 0)
        if current_round == total_rounds:
            final_path = Path(f"clients/client_{self.cid}/final_model.pt")
            self.model.save(final_path)
            print(f"CLIENT {self.cid}: 🎉 Final model saved")

        print(f"CLIENT {self.cid}: ✅ Round {current_round} complete!")
        print(f"{'='*70}\n")
        
        return trained_parameters, num_train_examples, {}

    def evaluate(self, parameters, config):
        """Evaluate model"""
        current_round = config.get("round", 0)
        print(f"CLIENT {self.cid}: 📉 Round {current_round} - Evaluating...")
        
        # Reset and load
        self._reset_model()
        self.set_parameters(parameters)

        metrics = self.model.val(
            data=str(self.yaml_path),
            split='val',
            workers=0,
            device=0,
            project=f'clients/client_{self.cid}/eval_run',
            name=f'round_{current_round}_eval',
            exist_ok=True,
            verbose=False
        )
        
        map50 = float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0.0
        map50_95 = float(metrics.box.map) if hasattr(metrics.box, 'map') else 0.0
        precision = float(metrics.box.p[0]) if hasattr(metrics.box, 'p') and len(metrics.box.p) > 0 else 0.0
        recall = float(metrics.box.r[0]) if hasattr(metrics.box, 'r') and len(metrics.box.r) > 0 else 0.0
        
        print(f"CLIENT {self.cid}: ✅ mAP50: {map50:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")

        metrics_dict = {
            "map50": map50,
            "map50-95": map50_95,
            "precision": precision,
            "recall": recall
        }
        
        fitness = float(metrics.fitness) if hasattr(metrics, 'fitness') else map50
        return fitness, self.num_val_examples, metrics_dict

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client for YOLOv8")
    parser.add_argument("--cid", type=int, required=True, help="Client ID")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    print("\n" + "="*70)
    print(f"🚀 STARTING FLOWER CLIENT {args.cid}")
    print("="*70 + "\n")
    
    yaml_path, num_val_examples = prepare_client_dataset(args.cid)
    client = FlowerClient(args.cid, yaml_path, num_val_examples)
    
    try:
        fl.client.start_numpy_client(
            server_address="127.0.0.1:8080",
            client=client
        )
    except Exception as e:
        print(f"\n❌ CLIENT {args.cid} CRASHED: {e}")
        raise
    
    print(f"\n✅ CLIENT {args.cid} FINISHED!")