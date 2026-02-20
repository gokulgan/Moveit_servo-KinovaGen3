#!/usr/bin/env python3
import sys
import os

# ==================== ADD THESE PATHS ====================
# Add MaskDINO and Detectron2 to Python path
sys.path.insert(0, "/workspace/MaskDINO")
sys.path.insert(0, "/workspace/detectron2")

# Verify paths
print("Python path:")
for p in sys.path[:5]:  # Show first 5 paths
    print(f"  {p}")
print()

# Now import the modules
try:
    import torch
    import detectron2
    from detectron2.utils.logger import setup_logger
    setup_logger()
    
    import maskdino  # This should work now
    
    print("✓ All imports successful!")
    print(f"Torch version: {torch.__version__}")
    print(f"Detectron2 available")
    print(f"MaskDINO available")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Current sys.path:")
    for p in sys.path:
        print(f"  {p}")
    sys.exit(1)
# ==================== END PATH FIX ====================

# Rest of your existing code continues here...
import numpy as np
import os, sys, yaml, json, cv2, datetime, argparse
from pathlib import Path
import warnings

import fiftyone as fo
from itertools import groupby
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

# Add after line 56 (after warnings.filterwarnings)

from detectron2.engine import HookBase

warnings.filterwarnings("ignore", category=RuntimeWarning)
class EarlyStoppingHook(HookBase):
    def __init__(self, patience=35, iterations_per_epoch=None):
        self.patience = patience
        self.eval_period = iterations_per_epoch  
        self.best_metric = -float('inf')
        self.counter = 0
        self.iterations_per_epoch = iterations_per_epoch
        
    def after_step(self):
      if self.trainer.iter % self.eval_period == 0:
        storage = self.trainer.storage
        
        # FIX: Extract value from tuple correctly
        try:
            # storage.latest() returns dict with (value, iteration) tuples
            latest = storage.latest()
            
            if "validation_loss" in latest:
                current_metric = -latest["validation_loss"][0]  # Extract first element
            elif "total_loss" in latest:
                current_metric = -latest["total_loss"][0]  # Extract first element
            else:
                print("⚠️ No loss metric found, skipping early stopping check")
                return
                
        except (KeyError, IndexError, TypeError) as e:
            print(f"⚠️ Error getting loss metric: {e}")
            return
        
        current_epoch = self.trainer.iter // self.iterations_per_epoch
        
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.counter = 0
            print(f"✓ Epoch {current_epoch}: Loss improved to {-current_metric:.4f}")
        else:
            self.counter += 1
            print(f"✗ Epoch {current_epoch}: No improvement ({self.counter}/{self.patience})")
            
        if self.counter >= self.patience:
            print(f"\n🛑 EARLY STOPPING at epoch {current_epoch}")
            print(f"No improvement for {self.patience} epochs")
            raise StopIteration
# ... rest of your existing code ...
#########################
### PROGRAM VARIABLES ###
#########################

DO_TRAIN = True
DO_TEST = True
RESUME = False
OUTPUT_DIR = "/data_private/outputs"
CONFIG_FILE = "configs/config_maskdino_swinL.yaml"
TEST_WEIGHTS = ""
INITIAL_WEIGHTS = None

def parse_ultrayolo_seg_format(txt_path, img_width=640, img_height=480):
    """
    Parse UltraYOLO segmentation format:
    Format: <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
    Coordinates are normalized (0-1)
    """
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    annotations = []
    for line in lines:
        parts = list(map(float, line.strip().split()))
        if len(parts) < 3:  # Need at least class_id + one point
            continue
        
        class_id = int(parts[0])
        points = parts[1:]
        
        # Convert normalized coordinates to pixel coordinates
        polygon = []
        for i in range(0, len(points), 2):
            if i + 1 < len(points):
                x = points[i] * img_width
                y = points[i + 1] * img_height
                polygon.extend([x, y])
        
        if len(polygon) >= 6:  # Need at least 3 points for a polygon
            # Calculate bounding box
            x_coords = polygon[0::2]
            y_coords = polygon[1::2]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            
            annotations.append({
                "category_id": class_id,
                "segmentation": [polygon],
                "bbox": bbox,
                "bbox_mode": BoxMode.XYWH_ABS
            })
    
    return annotations

def load_data_yaml(yaml_path):
    """Load and parse data.yaml file"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def load_ultrayolo_dataset(base_dir, data_yaml_path, split="train"):
    """
    Load UltraYOLO format dataset using data.yaml configuration
    """
    # Load data.yaml configuration
    data_config = load_data_yaml(data_yaml_path)
    base_dir = Path(base_dir)
    
    # Determine paths based on split
    if split == "train":
        image_dir = base_dir / "images" / "train"
        label_dir = base_dir / "labels" / "train"
    elif split == "val":
        image_dir = base_dir / "images" / "val"
        label_dir = base_dir / "labels" / "val"
    else:
        raise ValueError(f"Unknown split: {split}")
    
    # Get class names from data.yaml
    class_names = data_config.get('names', ['object'])
    num_classes = len(class_names)
    
    dataset_dicts = []
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(f"*{ext}")))
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    for img_idx, img_path in enumerate(image_files):
        # Find corresponding label file
        label_path = label_dir / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            print(f"Warning: No label file for {img_path.name}")
            continue
        
        # Read image to get dimensions
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            continue
        
        height, width = img.shape[:2]
        
        # Parse annotations
        try:
            annotations = parse_ultrayolo_seg_format(label_path, width, height)
        except Exception as e:
            print(f"Error parsing annotations for {label_path}: {e}")
            continue
        
        if annotations:
            record = {
                "file_name": str(img_path),
                "image_id": img_idx,
                "height": height,
                "width": width,
                "annotations": annotations
            }
            dataset_dicts.append(record)
        else:
            print(f"Warning: No valid annotations in {label_path}")
    
    print(f"Loaded {len(dataset_dicts)} samples for {split} split")
    return dataset_dicts, class_names, num_classes

def init_mask_dino(config_file, output_dir, dataset_train, dataset_val, 
                   class_names, num_classes, initial_weights=None):
    """Initialize MaskDINO configuration"""
    
    cfg = get_cfg()
    maskdino.add_maskdino_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.OUTPUT_DIR = output_dir
    
    # Set initial weights if provided
    if initial_weights is not None:
        cfg.MODEL.WEIGHTS = initial_weights
    
    # Register datasets
    DatasetCatalog.register("maskdino_train", lambda: dataset_train)
    MetadataCatalog.get("maskdino_train").set(thing_classes=class_names)
    
    DatasetCatalog.register("maskdino_val", lambda: dataset_val)
    MetadataCatalog.get("maskdino_val").set(thing_classes=class_names)
    
    # Training configuration
    cfg.DATASETS.TRAIN = ("maskdino_train",)
    cfg.DATASETS.TEST = ("maskdino_val",)
    
    # Model configuration
    if hasattr(cfg.MODEL, 'MASK_FORMER'):
        cfg.MODEL.MASK_FORMER.NUM_CLASSES = num_classes
    
    if hasattr(cfg.MODEL, 'SEM_SEG_HEAD'):
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    
    if hasattr(cfg.MODEL, 'ROI_HEADS'):
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    elif hasattr(cfg.MODEL, 'ROI_BOX_HEAD'):  # Alternative name
        cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES = num_classes

    # Calculate iterations based on dataset size and epochs
    num_images = len(dataset_train)
    batch_size = cfg.SOLVER.IMS_PER_BATCH
    
    # Ensure batch size is reasonable
    if batch_size > 4:  # Reduce for memory constraints
        cfg.SOLVER.IMS_PER_BATCH = 2
        batch_size = 2
    
    # Calculate max iterations
    iterations_per_epoch = max(1, num_images // batch_size)
    cfg.SOLVER.MAX_ITER = iterations_per_epoch * args.epochs
    
    # Learning rate configuration
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.WARMUP_ITERS = min(1000, cfg.SOLVER.MAX_ITER // 10)
    
    # LR schedule - step decay
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (
        int(cfg.SOLVER.MAX_ITER * 0.6),
        int(cfg.SOLVER.MAX_ITER * 0.8)
    )
    
    # Checkpoint saving
    #cfg.SOLVER.CHECKPOINT_PERIOD = iterations_per_epoch  # Save once per epoch
    
    # Save configuration
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/config.yaml", "w") as f:
        f.write(cfg.dump())
    
    # Save dataset info
    dataset_info = {
        "num_train_samples": len(dataset_train),
        "num_val_samples": len(dataset_val),
        "class_names": class_names,
        "num_classes": num_classes,
        "training_iterations": cfg.SOLVER.MAX_ITER,
        "batch_size": batch_size
    }
    
    with open(f"{output_dir}/dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    print(f"Training samples: {len(dataset_train)}")
    print(f"Validation samples: {len(dataset_val)}")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Batch size: {batch_size}")
    print(f"Total iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"Epochs: {args.epochs}")
    #print(f"Checkpoint frequency: every {cfg.SOLVER.CHECKPOINT_PERIOD} iterations")
    print("="*50 + "\n")
    
    return cfg

def train_mask_dino(cfg, resume=False, iterations_per_epoch=60, patience=5):
    """Train MaskDINO model with early stopping"""
    from train_net import Trainer
    
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume)
    
    # Add early stopping hook
    early_stop_hook = EarlyStoppingHook(
        patience=patience,  # Stop after 35 epochs without improvement
        iterations_per_epoch=iterations_per_epoch  # Evaluate every epoch
    )
    trainer.register_hooks([early_stop_hook])
    
    # Train the model
    try:
        trainer.train()
    except StopIteration:
        print("Training stopped early due to no improvement")

def evaluate_model(cfg, dataset_val, output_dir):
    """Evaluate model on validation set"""
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    
    metadata = MetadataCatalog.get("maskdino_val")
    
    results = {
        "total_samples": len(dataset_val),
        "predictions": []
    }
    
    for i, d in enumerate(dataset_val):
        try:
            img = cv2.imread(d["file_name"])
            if img is None:
                print(f"Could not read image: {d['file_name']}")
                continue
                
            outputs = predictor(img)
            instances = outputs["instances"].to("cpu")
            
            # Count predictions
            num_predictions = len(instances)
            
            # Calculate average confidence
            if num_predictions > 0:
                avg_confidence = instances.scores.mean().item()
            else:
                avg_confidence = 0.0
            
            results["predictions"].append({
                "image_id": d["image_id"],
                "num_predictions": num_predictions,
                "avg_confidence": avg_confidence
            })
            
            # Visualize first few samples
            if i < 5:
                v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.8)
                out = v.draw_instance_predictions(instances)
                vis_path = os.path.join(output_dir, f"prediction_{i}.jpg")
                cv2.imwrite(vis_path, out.get_image()[:, :, ::-1])
                print(f"Saved visualization: {vis_path}")
                
        except Exception as e:
            print(f"Error processing image {d['file_name']}: {e}")
            continue
    
    # Save evaluation results
    eval_path = os.path.join(output_dir, "evaluation_results.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation complete. Results saved to {eval_path}")
    
    # Print summary
    total_predictions = sum([p["num_predictions"] for p in results["predictions"]])
    avg_predictions = total_predictions / len(results["predictions"])
    
    print(f"\nEvaluation Summary:")
    print(f"Total validation samples: {len(results['predictions'])}")
    print(f"Total predictions made: {total_predictions}")
    print(f"Average predictions per image: {avg_predictions:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Train Mask Dino on UltraYOLO Dataset',
        description='Train instance segmentation model on UltraYOLO format dataset',
        epilog='Note: Expects data.yaml and UltraYOLO format .txt files'
    )
    
    parser.add_argument('--config', type=str, default=CONFIG_FILE,
                       help='Path to MaskDINO config file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Base directory containing images/ and labels/ folders')
    parser.add_argument('--data_yaml', type=str, required=True,
                       help='Path to data.yaml file')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                       help='Output directory for results and checkpoints')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    
    print('='*60)
    print('MaskDINO Training Script')
    print('='*60)
    print(f'GPU available: {torch.cuda.is_available()}')
    print(f'PyTorch version: {torch.__version__}')
    print(f'Detectron2 version: {detectron2.__version__}')
    print('='*60)
    
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    
    args = parser.parse_args()
    
    # Create output directory
    if RESUME:
        output_dirs = [d for d in os.listdir(args.output_dir) 
                      if os.path.isdir(os.path.join(args.output_dir, d))]
        if output_dirs:
            output_dir = os.path.join(args.output_dir, sorted(output_dirs)[-1])
            print(f"Resuming from: {output_dir}")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, f"maskdino_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    # Load datasets
    print("\nLoading datasets...")
    print(f"Data directory: {args.data_dir}")
    print(f"Data YAML: {args.data_yaml}")
    
    dataset_train, train_class_names, num_classes = load_ultrayolo_dataset(
        args.data_dir, args.data_yaml, split="train"
    )
    
    dataset_val, val_class_names, _ = load_ultrayolo_dataset(
        args.data_dir, args.data_yaml, split="val"
    )
    
    if not dataset_train:
        print("ERROR: No training data loaded!")
        sys.exit(1)
    
    if not dataset_val:
        print("WARNING: No validation data loaded!")
        # Use training data for validation if needed
        dataset_val = dataset_train[:min(10, len(dataset_train))]
    
    # Initialize and train model
    print("\nInitializing MaskDINO...")
    cfg = init_mask_dino(
        config_file=args.config,
        output_dir=output_dir,
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        class_names=train_class_names,
        num_classes=num_classes,
        initial_weights=INITIAL_WEIGHTS
    )
    
    if DO_TRAIN:
      print("\nStarting training...")
      iterations_per_epoch = max(1, len(dataset_train) // cfg.SOLVER.IMS_PER_BATCH)
      train_mask_dino(cfg, RESUME, iterations_per_epoch=iterations_per_epoch, patience=35)
      
      TEST_WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # ← MISSING!
      print(f"Training complete. Model saved to: {TEST_WEIGHTS}")
    
    if DO_TEST and TEST_WEIGHTS and os.path.exists(TEST_WEIGHTS):
        print("\nEvaluating model on validation set...")
        evaluate_model(cfg, dataset_val, output_dir)
    elif DO_TEST:
        print("\nSkipping evaluation - no trained model found")
    
    print("\n" + "="*60)
    print("Training script completed successfully!")
    print(f"Results saved in: {output_dir}")
    print("="*60)












