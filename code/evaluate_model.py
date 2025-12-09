"""
Evaluate the best model performance
Load saved best_model and evaluate all metrics on test set
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data_loader import ARGDataLoader
from modules import CML
from utils import evaluate
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ==================== Configuration ====================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 32

# Model parameters (must match training configuration)
X_dim = 128
G_dim = 128
z_dim = 256
EMS_input_dim = 1280
antibiotic_count = 15
mechanism_count = 6
transfer_count = 2

# Model file path (modify as needed)
MODEL_PATH = './res/best_model.pth'  

print("=" * 80)
print("CDM-ARG Best Model Evaluation")
print("=" * 80)
print(f"Device: {device}")
print(f"Model Path: {MODEL_PATH}")
print(f"Batch Size: {batch_size}")
print("=" * 80)

# ==================== Load Data ====================
print("\n[1/4] Loading test data...")
dataloader = ARGDataLoader()
test_dataloader = dataloader.load_test_dataSet(batch_size)
print(f" Test batches: {len(test_dataloader)}")
print(f" Total samples: ~{len(test_dataloader) * batch_size}")

# ==================== Load Model ====================
print("\n[2/4] Loading model...")
model = CML(X_dim, G_dim, z_dim, EMS_input_dim, antibiotic_count, mechanism_count, transfer_count)
model = model.to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f" Successfully loaded model: {MODEL_PATH}")
except FileNotFoundError:
    print(f"✗ Error: Model file not found {MODEL_PATH}")
    print("Please verify the model file path is correct")
    exit(1)
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

# Load ProteinBERT (if used)
from protein_bert_pytorch import ProteinBERT
modelProtein = ProteinBERT(
    num_tokens=23,
    num_annotation=1,
    dim=512,
    dim_global=256,
    depth=6,
    narrow_conv_kernel=9,
    wide_conv_kernel=9,
    wide_conv_dilation=5,
    attn_heads=8,
    attn_dim_head=64
).to(device)
print(" ProteinBERT loaded")

# ==================== Evaluate Model ====================
print("\n[3/4] Evaluating on test set...")
model.eval()
modelProtein.eval()

# Initialize storage
test_transfer_pred = np.empty(shape=[0, transfer_count])
test_transfer_label = np.array([])
test_mechanism_pred = np.empty(shape=[0, mechanism_count])
test_mechanism_label = np.array([])
test_antibiotic_pred = np.empty(shape=[0, antibiotic_count])
test_antibiotic_label = np.array([])

# Statistics
total_samples = 0
batch_count = 0

with torch.no_grad():
    for index, (seq, seq_map, transfer_label, mechanism_label, antibiotic_label) in enumerate(test_dataloader):
        batch_count += 1
        batch_size_actual = seq_map.size(0)
        total_samples += batch_size_actual

        # Data preparation
        seq_map = seq_map.view(-1, 1, 1576, 23).to(device)
        transfer_label = transfer_label.to(device)
        mechanism_label = mechanism_label.to(device)
        antibiotic_label = antibiotic_label.to(device)

        # ProteinBERT processing
        adjusted_input = seq_map.squeeze(1)
        index_seq_map = torch.argmax(adjusted_input, dim=-1).to(device)
        mask = torch.ones_like(index_seq_map, dtype=torch.bool).to(device)
        annotation = torch.randint(0, 1, (index_seq_map.size(0), 1)).float().to(device)
        seq_logits, _ = modelProtein(index_seq_map, annotation, mask=mask)
        expanded_tensor = seq_logits.unsqueeze(1)

        # Model prediction
        antibiotic_output, mechanism_output, transfer_output, _ = model.forward(
            expanded_tensor,
            antibiotic_label.view(-1, 1),
            mechanism_label.view(-1, 1),
            transfer_label.view(-1, 1),
            antibiotic_count, mechanism_count, transfer_count
        )

        # Convert to numpy and accumulate
        transfer_output = transfer_output.cpu().detach().numpy()
        test_transfer_pred = np.append(test_transfer_pred, transfer_output, axis=0)
        test_transfer_label = np.concatenate((test_transfer_label, transfer_label.cpu().numpy()))

        mechanism_output = mechanism_output.cpu().detach().numpy()
        test_mechanism_pred = np.append(test_mechanism_pred, mechanism_output, axis=0)
        test_mechanism_label = np.concatenate((test_mechanism_label, mechanism_label.cpu().numpy()))

        antibiotic_output = antibiotic_output.cpu().detach().numpy()
        test_antibiotic_pred = np.append(test_antibiotic_pred, antibiotic_output, axis=0)
        test_antibiotic_label = np.concatenate((test_antibiotic_label, antibiotic_label.cpu().numpy()))

        # Progress display
        if (index + 1) % 10 == 0 or (index + 1) == len(test_dataloader):
            print(f"  Progress: {index + 1}/{len(test_dataloader)} batches ({total_samples} samples)", end='\r')

print(f"\n Evaluation completed, total samples: {total_samples}")

# ==================== Calculate and Display Results ====================
print("\n[4/4] Calculating evaluation metrics...")
print("=" * 80)
print("Test Set Performance Results")
print("=" * 80)

# Transfer task
print("\n 1. Transfer Task (Transferability Prediction)")
print("-" * 80)
acc, macro_p, macro_r, macro_f1, auc, aucpr = evaluate(test_transfer_pred, test_transfer_label, transfer_count)
print(f"  Accuracy:     {acc:.4f}")
print(f"  Precision:    {macro_p:.4f}")
print(f"  Recall:       {macro_r:.4f}")
print(f"  F1-Score:     {macro_f1:.4f}")
print(f"  AUC:          {auc:.4f}")
print(f"  AUCPR:        {aucpr:.4f}")

# Detailed classification report
transfer_pred_labels = np.argmax(test_transfer_pred, axis=1)
print("\n  Detailed Classification Report:")
print(classification_report(test_transfer_label, transfer_pred_labels,
                          target_names=['Non-transferable', 'Transferable'],
                          digits=4))

# Mechanism task
print("\n 2. Mechanism Task (Resistance Mechanism Prediction)")
print("-" * 80)
acc, macro_p, macro_r, macro_f1, auc, aucpr = evaluate(test_mechanism_pred, test_mechanism_label, mechanism_count)
print(f"  Accuracy:     {acc:.4f}")
print(f"  Precision:    {macro_p:.4f}")
print(f"  Recall:       {macro_r:.4f}")
print(f"  F1-Score:     {macro_f1:.4f}")
print(f"  AUC:          {auc:.4f}")
print(f"  AUCPR:        {aucpr:.4f}")

# Detailed classification report
mechanism_pred_labels = np.argmax(test_mechanism_pred, axis=1)
# Get unique classes present in test set
unique_classes = sorted(np.unique(test_mechanism_label).astype(int))
mechanism_names = [f'Mechanism_{i}' for i in unique_classes]
print("\n  Detailed Classification Report:")
print(classification_report(test_mechanism_label, mechanism_pred_labels,
                          labels=unique_classes,
                          target_names=mechanism_names,
                          digits=4, zero_division=0))

# Class distribution
unique, counts = np.unique(test_mechanism_label, return_counts=True)
print("\n  Test Set Class Distribution:")
for cls, count in zip(unique, counts):
    print(f"    Class {int(cls)}: {count} samples ({count/len(test_mechanism_label)*100:.1f}%)")

# Antibiotic task
print("\n3. Antibiotic Task (Antibiotic Class Prediction)")
print("-" * 80)
acc, macro_p, macro_r, macro_f1, auc, aucpr = evaluate(test_antibiotic_pred, test_antibiotic_label, antibiotic_count)
print(f"  Accuracy:     {acc:.4f}")
print(f"  Precision:    {macro_p:.4f}")
print(f"  Recall:       {macro_r:.4f}")
print(f"  F1-Score:     {macro_f1:.4f}")
print(f"  AUC:          {auc:.4f}")
print(f"  AUCPR:        {aucpr:.4f}")

# Detailed classification report
antibiotic_pred_labels = np.argmax(test_antibiotic_pred, axis=1)
# Get unique classes present in test set
unique_classes_ab = sorted(np.unique(test_antibiotic_label).astype(int))
antibiotic_names = [f'Antibiotic_{i}' for i in unique_classes_ab]
print("\n  Detailed Classification Report:")
print(classification_report(test_antibiotic_label, antibiotic_pred_labels,
                          labels=unique_classes_ab,
                          target_names=antibiotic_names,
                          digits=4, zero_division=0))

# Class distribution (show top 10)
unique, counts = np.unique(test_antibiotic_label, return_counts=True)
print("\n  Test Set Class Distribution (Top 10):")
for cls, count in zip(unique[:10], counts[:10]):
    print(f"    Class {int(cls)}: {count} samples ({count/len(test_antibiotic_label)*100:.1f}%)")
if len(unique) > 10:
    print(f"    ... and {len(unique)-10} more classes")

# ==================== Performance Summary ====================
print("\n" + "=" * 80)
print("Performance Summary")
print("=" * 80)

# Create results DataFrame
results_df = pd.DataFrame({
    'Task': ['Transfer', 'Mechanism', 'Antibiotic'],
    'Accuracy': [
        evaluate(test_transfer_pred, test_transfer_label, transfer_count)[0],
        evaluate(test_mechanism_pred, test_mechanism_label, mechanism_count)[0],
        evaluate(test_antibiotic_pred, test_antibiotic_label, antibiotic_count)[0]
    ],
    'Precision': [
        evaluate(test_transfer_pred, test_transfer_label, transfer_count)[1],
        evaluate(test_mechanism_pred, test_mechanism_label, mechanism_count)[1],
        evaluate(test_antibiotic_pred, test_antibiotic_label, antibiotic_count)[1]
    ],
    'Recall': [
        evaluate(test_transfer_pred, test_transfer_label, transfer_count)[2],
        evaluate(test_mechanism_pred, test_mechanism_label, mechanism_count)[2],
        evaluate(test_antibiotic_pred, test_antibiotic_label, antibiotic_count)[2]
    ],
    'F1-Score': [
        evaluate(test_transfer_pred, test_transfer_label, transfer_count)[3],
        evaluate(test_mechanism_pred, test_mechanism_label, mechanism_count)[3],
        evaluate(test_antibiotic_pred, test_antibiotic_label, antibiotic_count)[3]
    ],
    'AUC': [
        evaluate(test_transfer_pred, test_transfer_label, transfer_count)[4],
        evaluate(test_mechanism_pred, test_mechanism_label, mechanism_count)[4],
        evaluate(test_antibiotic_pred, test_antibiotic_label, antibiotic_count)[4]
    ],
    'AUCPR': [
        evaluate(test_transfer_pred, test_transfer_label, transfer_count)[5],
        evaluate(test_mechanism_pred, test_mechanism_label, mechanism_count)[5],
        evaluate(test_antibiotic_pred, test_antibiotic_label, antibiotic_count)[5]
    ]
})

print(results_df.to_string(index=False))

# ==================== Save Results ====================
print("\n" + "=" * 80)
print("Saving Evaluation Results")
print("=" * 80)

# Save to CSV
results_df.to_csv('./res/evaluation_results.csv', index=False)
print(" Performance metrics saved to: ./res/evaluation_results.csv")

# Save detailed predictions
predictions_df = pd.DataFrame({
    'transfer_true': test_transfer_label,
    'transfer_pred': np.argmax(test_transfer_pred, axis=1),
    'mechanism_true': test_mechanism_label,
    'mechanism_pred': np.argmax(test_mechanism_pred, axis=1),
    'antibiotic_true': test_antibiotic_label,
    'antibiotic_pred': np.argmax(test_antibiotic_pred, axis=1)
})
predictions_df.to_csv('./res/detailed_predictions.csv', index=False)
print(" Detailed predictions saved to: ./res/detailed_predictions.csv")

# Save prediction probabilities
np.savez('./res/prediction_probabilities.npz',
         transfer_prob=test_transfer_pred,
         mechanism_prob=test_mechanism_pred,
         antibiotic_prob=test_antibiotic_pred)
print(" Prediction probabilities saved to: ./res/prediction_probabilities.npz")

print("\n" + "=" * 80)
print("Evaluation Completed!")
print("=" * 80)
