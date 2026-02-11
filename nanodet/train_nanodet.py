"""
NanoDet-m Training Script for Barcode Detection
================================================

This script sets up and trains NanoDet-m model on the augmented barcode dataset.
NanoDet is a FCOS-style one-stage anchor-free object detection model.

Repository: https://github.com/RangiLyu/nanodet

Requirements:
- PyTorch >= 1.10.0
- pytorch-lightning
- nanodet library

Installation:
    pip install nanodet

Or install from source:
    git clone https://github.com/RangiLyu/nanodet.git
    cd nanodet
    pip install -e .

Usage:
    python train_nanodet.py
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def check_nanodet_installation():
    """Check if nanodet is installed, if not provide installation instructions."""
    try:
        import nanodet
        print(f"✓ NanoDet version: {nanodet.__version__}")
        return True
    except ImportError:
        print("✗ NanoDet is not installed.")
        return False


def install_nanodet():
    """Install nanodet from PyPI or source."""
    print("\nInstalling NanoDet...")
    
    # Try pip install first
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nanodet"])
        print("✓ NanoDet installed successfully via pip")
        return True
    except subprocess.CalledProcessError:
        print("pip install failed, trying source installation...")
    
    # Clone and install from source
    nanodet_dir = Path(__file__).parent / "nanodet_src"
    if not nanodet_dir.exists():
        subprocess.run([
            "git", "clone", 
            "https://github.com/RangiLyu/nanodet.git",
            str(nanodet_dir)
        ])
    
    os.chdir(nanodet_dir)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
    os.chdir(Path(__file__).parent)
    print("✓ NanoDet installed from source")
    return True


def get_config_path():
    """Get the config file path."""
    config_path = Path(__file__).parent / "config_nanodet_barcode.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return str(config_path)


def verify_dataset():
    """Verify that the dataset exists and is properly formatted."""
    base_dir = Path(__file__).parent.parent
    
    required_paths = [
        base_dir / "augmented_barcodes" / "train" / "_annotations.coco.json",
        base_dir / "augmented_barcodes" / "valid" / "_annotations.coco.json",
        base_dir / "augmented_barcodes" / "test" / "_annotations.coco.json",
    ]
    
    print("\nVerifying dataset...")
    all_exist = True
    for path in required_paths:
        if path.exists():
            print(f"  ✓ {path.relative_to(base_dir)}")
        else:
            print(f"  ✗ {path.relative_to(base_dir)} - NOT FOUND")
            all_exist = False
    
    if all_exist:
        # Load and check annotations
        import json
        train_ann_path = required_paths[0]
        with open(train_ann_path, 'r') as f:
            data = json.load(f)
        
        categories = data.get('categories', [])
        images = data.get('images', [])
        annotations = data.get('annotations', [])
        
        print(f"\nDataset Statistics (train):")
        print(f"  - Categories: {[c['name'] for c in categories]}")
        print(f"  - Number of images: {len(images)}")
        print(f"  - Number of annotations: {len(annotations)}")
    
    return all_exist


def train_nanodet(config_path: str, resume: str = None):
    """
    Train NanoDet model using the specified config.
    
    Args:
        config_path: Path to the YAML configuration file
        resume: Path to checkpoint to resume from (optional)
    """
    try:
        from nanodet.util import cfg, load_config, Logger
        from nanodet.trainer import build_trainer
        from nanodet.data.collate import naive_collate
        from nanodet.data.dataset import build_dataset
        from nanodet.model import build_model
        
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
        import torch
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nPlease install NanoDet first:")
        print("  pip install nanodet")
        print("  # or")
        print("  git clone https://github.com/RangiLyu/nanodet.git")
        print("  cd nanodet && pip install -e .")
        return
    
    # Load configuration
    load_config(cfg, config_path)
    
    # Create save directory
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = Logger(local_rank=-1, save_dir=str(save_dir), use_tensorboard=True)
    
    # Log config
    logger.log(f"Config:\n{cfg}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model(cfg.model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Build dataset
    print("\nLoading datasets...")
    train_dataset = build_dataset(cfg.data.train, 'train')
    val_dataset = build_dataset(cfg.data.val, 'val')
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Build dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=True,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=False
    )
    
    # Build trainer
    print("\nInitializing trainer...")
    trainer_module = build_trainer(model, cfg)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=str(save_dir),
            filename='nanodet_barcode_{epoch:03d}_{mAP:.4f}',
            monitor='mAP',
            mode='max',
            save_top_k=3,
            save_last=True
        ),
        TQDMProgressBar(refresh_rate=1)
    ]
    
    # Setup PyTorch Lightning trainer
    pl_trainer = pl.Trainer(
        default_root_dir=str(save_dir),
        max_epochs=cfg.schedule.total_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=cfg.device.gpu_ids if torch.cuda.is_available() else 1,
        callbacks=callbacks,
        gradient_clip_val=cfg.grad_clip,
        val_check_interval=cfg.schedule.val_intervals,
        log_every_n_steps=cfg.log.interval,
        precision=cfg.device.precision
    )
    
    # Start training
    print("\n" + "="*50)
    print("Starting NanoDet Training")
    print("="*50)
    
    if resume:
        pl_trainer.fit(trainer_module, train_loader, val_loader, ckpt_path=resume)
    else:
        pl_trainer.fit(trainer_module, train_loader, val_loader)
    
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Models saved to: {save_dir}")
    print("="*50)


def train_with_cli(config_path: str):
    """
    Alternative training method using NanoDet's built-in CLI.
    This is the recommended approach as it handles all configurations automatically.
    """
    # First, check if nanodet tools exist
    nanodet_src = Path(__file__).parent / "nanodet_src"
    
    if nanodet_src.exists():
        train_script = nanodet_src / "tools" / "train.py"
        if train_script.exists():
            print(f"\nUsing NanoDet CLI: {train_script}")
            subprocess.run([
                sys.executable, 
                str(train_script), 
                config_path
            ])
            return
    
    # Try using nanodet module directly
    print("\nUsing nanodet module for training...")
    subprocess.run([
        sys.executable, "-m", "nanodet.tools.train",
        config_path
    ])


def main():
    parser = argparse.ArgumentParser(description='Train NanoDet-m on barcode dataset')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: config_nanodet_barcode.yml)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    parser.add_argument('--install', action='store_true',
                        help='Install NanoDet if not present')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify dataset without training')
    parser.add_argument('--use-cli', action='store_true',
                        help='Use NanoDet CLI for training (recommended)')
    
    args = parser.parse_args()
    
    print("="*50)
    print("NanoDet-m Barcode Detection Training")
    print("="*50)
    
    # Check/install nanodet
    if not check_nanodet_installation():
        if args.install:
            install_nanodet()
        else:
            print("\nRun with --install to automatically install NanoDet")
            print("Or manually install:")
            print("  pip install nanodet")
            return
    
    # Verify dataset
    if not verify_dataset():
        print("\n✗ Dataset verification failed!")
        return
    
    if args.verify_only:
        print("\n✓ Dataset verification passed!")
        return
    
    # Get config path
    config_path = args.config or get_config_path()
    print(f"\nUsing config: {config_path}")
    
    # Train
    if args.use_cli:
        train_with_cli(config_path)
    else:
        train_nanodet(config_path, args.resume)


if __name__ == "__main__":
    main()
