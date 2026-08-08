"""
Background Model Fine-Tuner Script (scripts/run_background_trainer.py)
Executes deep 20-epoch local PyTorch MPS fine-tuning & subject training for roadrunner and human portrait anatomy.
"""

import sys
import os
import time

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.trainer_agent import SelfOptimizingVisualTrainer

def main():
    print("============================================================")
    print("🚀 SenaAIgent Autonomous Background Fine-Tuning Agent Started")
    print("============================================================")
    
    trainer = SelfOptimizingVisualTrainer()

    # 1. Train Roadrunner Species Weights
    print("\n[Phase 1/3] Training Roadrunner Species Weights...")
    res_roadrunner = trainer.fetch_and_train_subject("roadrunner")
    print(f"✅ Roadrunner Training Complete: Score {res_roadrunner.get('visual_match_score')}% | Checkpoint: {res_roadrunner.get('fine_tuning', {}).get('checkpoint_file')}")

    # 2. Train Human Anatomy & Portrait Structure Weights
    print("\n[Phase 2/3] Training Human Anatomy & Portrait Structure Weights...")
    res_human = trainer.fetch_and_train_subject("human_anatomy_portrait")
    print(f"✅ Human Anatomy Training Complete: Score {res_human.get('visual_match_score')}% | Checkpoint: {res_human.get('fine_tuning', {}).get('checkpoint_file')}")

    # 3. Full Deep 20-Epoch Checkpoint Fine-Tuning
    print("\n[Phase 3/3] Running Deep 20-Epoch Checkpoint Optimization...")
    res_full = trainer.run_full_model_training(training_dir="static/training_data", epochs=20)
    print(f"🎉 Full Local Fine-Tuning Complete!")
    print(f"   - Epochs: {res_full.get('epochs_completed')}")
    print(f"   - Final Loss: {res_full.get('final_loss')}")
    print(f"   - Checkpoint Saved: {res_full.get('checkpoint_file')}")
    print("============================================================")

if __name__ == "__main__":
    main()
