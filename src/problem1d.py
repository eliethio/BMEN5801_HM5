import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

# Ensure output directories exist
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)

# Load dataset
df = pd.read_csv('data/placental_geneexp.csv')

features = ['LGALS14', 'MGST1', 'ATXN3', 'SSU72', 'ICAM2']
X = df[features]
y = df['SMOKER']

# 70:30 train:test split (stratified to maintain smoker/non-smoker ratio, random_state=10)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=10, stratify=y
)

print(f"Total samples: {len(df)} | Train samples: {len(X_train)} | Test samples: {len(X_test)}")
print(f"Test set class counts -> Non-Smoker (0): {(y_test == 0).sum()}, Smoker (1): {(y_test == 1).sum()}\n")

# Sweep var_smoothing across orders of magnitude (default in sklearn is 1e-9)
var_smoothing_values = np.logspace(-12, -1, 12)

results = []
for vs in var_smoothing_values:
    gnb = GaussianNB(var_smoothing=vs)
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    results.append({
        'var_smoothing': vs,
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'Accuracy': round(acc, 4),
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn
    })

results_df = pd.DataFrame(results)

# Save summary table to results/tables
table_path = 'results/tables/problem1d_gnb_var_smoothing.csv'
results_df.to_csv(table_path, index=False)
print(f"Results table saved successfully to {table_path}\n")
print("Gaussian Naive Bayes Performance Across var_smoothing (70:30 Split):")
print(results_df.to_string(index=False))

# Create line plot showing Precision and Recall vs var_smoothing
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    results_df['var_smoothing'], results_df['Precision'],
    marker='o', linewidth=2.2, markersize=7, color='#1f77b4', label='Precision'
)
ax.plot(
    results_df['var_smoothing'], results_df['Recall'],
    marker='s', linestyle='--', linewidth=2.2, markersize=7, color='#d62728', label='Recall'
)

ax.set_xscale('log')
ax.set_ylim(-0.05, 1.05)
ax.set_title('Gaussian Naïve Bayes: Precision & Recall vs. var_smoothing (70:30 Split)', fontsize=12, fontweight='bold')
ax.set_xlabel('var_smoothing (log scale)', fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.legend(loc='best', fontsize=10)
ax.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
fig_path = 'results/figures/problem1d_gnb_precision_recall.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved successfully to {fig_path}")
plt.show()