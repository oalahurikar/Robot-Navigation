#!/usr/bin/env python3
"""
Quick test script to compare baseline vs optimized configuration
"""

import torch
import numpy as np
from pathlib import Path
from core.data_generation import load_training_data
from core.pytorch_network import RobotNavigationNet, RobotNavigationTrainer, create_data_loaders, load_config

def test_config(config_file: str, model_name: str):
    """Test a configuration and report results"""
    
    print(f"\n{'='*70}")
    print(f"🧪 Testing: {model_name}")
    print(f"   Config: {config_file}")
    print(f"{'='*70}\n")
    
    # Load config
    config = load_config(config_file)
    
    # Load data
    X_train, y_train, _ = load_training_data('data/raw/large_training_dataset.npz')
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train,
        batch_size=config['training']['batch_size'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio']
    )
    
    # Create model
    model = RobotNavigationNet(
        input_size=config['model']['input_size'],
        perception_size=config['model']['perception_size'],
        history_size=config['model']['history_size'],
        hidden1_size=config['model']['hidden1_size'],
        hidden2_size=config['model']['hidden2_size'],
        output_size=config['model']['output_size'],
        dropout_rate=config['model']['dropout_rate']
    )
    
    # Create trainer with optimized settings
    trainer = RobotNavigationTrainer(
        model=model,
        learning_rate=config['training']['learning_rate']
    )
    
    # Display configuration
    print("⚙️  Configuration:")
    print(f"   Dropout: {config['model']['dropout_rate']}")
    print(f"   Learning Rate: {config['training']['learning_rate']}")
    print(f"   Batch Size: {config['training']['batch_size']}")
    print(f"   Weight Decay: {config['model'].get('weight_decay', 0.0)}")
    print(f"   LR Scheduler: {config['training']['lr_scheduler'].get('enabled', False)}")
    
    # Train model
    print(f"\n🚀 Starting training...")
    history = trainer.train(
        train_loader, 
        val_loader,
        epochs=config['training']['epochs'],
        early_stopping_patience=config['training']['early_stopping']['patience'],
        verbose=True
    )
    
    # Evaluate on test set
    test_loss, test_acc = trainer.evaluate(test_loader)
    
    # Report results
    final_train_acc = history['train_accuracies'][-1]
    final_val_acc = history['val_accuracies'][-1]
    overfitting_gap = final_train_acc - final_val_acc
    
    print(f"\n📊 FINAL RESULTS ({model_name}):")
    print(f"{'='*70}")
    print(f"   Training Accuracy:   {final_train_acc:.2f}%")
    print(f"   Validation Accuracy: {final_val_acc:.2f}%")
    print(f"   Test Accuracy:       {test_acc:.2f}%")
    print(f"   Overfitting Gap:     {overfitting_gap:.2f}%")
    print(f"   Status: {'✅ Improved!' if overfitting_gap < 3 else '⚠️  Still overfitting'}")
    
    return {
        'train_acc': final_train_acc,
        'val_acc': final_val_acc,
        'test_acc': test_acc,
        'overfitting': overfitting_gap
    }

if __name__ == "__main__":
    print("🎯 Hyperparameter Optimization Experiment")
    print("Testing optimized configuration vs baseline\n")
    
    # Test optimized config
    results_optimized = test_config(
        'configs/nn_config_optimized.yaml',
        'OPTIMIZED'
    )
    
    print("\n" + "="*70)
    print("🎊 EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"\n📈 Results Summary:")
    print(f"   Validation Accuracy: {results_optimized['val_acc']:.2f}%")
    print(f"   Test Accuracy:       {results_optimized['test_acc']:.2f}%")
    print(f"   Overfitting Gap:     {results_optimized['overfitting']:.2f}%")
    
    if results_optimized['val_acc'] > 78:
        print("\n✅ SUCCESS! Achieved 78%+ validation accuracy!")
    elif results_optimized['val_acc'] > 76.7:
        print("\n✅ IMPROVEMENT! Better than baseline 76.7%")
    else:
        print("\n⚠️  Try more aggressive regularization")
