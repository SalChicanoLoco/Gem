"""
Deep Anatomical & Portrait Trainer (scripts/deep_model_trainer.py)
Runs 50-iteration fine-tuning and prompt weight optimization over human portrait anatomy, clothing, and environment textures.
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.trainer_agent import SelfOptimizingVisualTrainer

def main():
    print("============================================================")
    print("🔥 SenaAIgent Deep Anatomical Model Trainer & Evaluator Started")
    print("============================================================")
    
    trainer = SelfOptimizingVisualTrainer()

    subjects = [
        ("latina_portrait_new_mexico", "Single latina woman in a bikini in New Mexico desert landscape"),
        ("roadrunner_desert", "Greater Roadrunner bird in desert habitat with cacti"),
        ("portrait_anatomy_symmetry", "Photorealistic portrait with natural skin texture and anatomical accuracy"),
    ]

    for name, desc in subjects:
        print(f"\n[Training Task] Training subject: '{name}'...")
        res = trainer.fetch_and_train_subject(name)
        print(f"  - Score Achieved: {res.get('visual_match_score')}%")
        print(f"  - Fine-Tuning Status: {res.get('fine_tuning', {}).get('message')}")

    print("\n[Deep Optimization] Running 30-Epoch Checkpoint Fine-Tuning...")
    full_res = trainer.run_full_model_training(epochs=30)
    print(f"\n🎉 Deep Model Training Complete!")
    print(f"   - Epochs: {full_res.get('epochs_completed')}")
    print(f"   - Loss: {full_res.get('final_loss')}")
    print(f"   - Checkpoint: {full_res.get('checkpoint_file')}")
    print("============================================================")

if __name__ == "__main__":
    main()
