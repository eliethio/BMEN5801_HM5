import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedKFold, cross_val_score

# Ensure output directories exist
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)

# 1. Load dataset
df = pd.read_csv('data/enzymes.csv')

feature_cols = [
    'MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex',
    'qed', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons'
]
target_col = 'EC3'

X = df[feature_cols]
y = df[target_col].astype(int)

print(f"Total enzyme samples: {len(df)}")
print(f"Features ({len(feature_cols)}): {feature_cols}")
print(f"Target ('{target_col}') -> Non-Hydrolase (0): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%), "
      f"Hydrolase [EC 3] (1): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)\n")

# ---------------------------------------------------------
# Problem 2(a): kNN Hyperparameter Sweep (Repeated 5-Fold CV)
# ---------------------------------------------------------
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=10)

# Sweep values of k from 1 to 50
k_values = list(range(1, 51))
cv_results = []

for k in k_values:
    # Standardize features within each CV fold so all 9 molecular descriptors contribute equally to Euclidean distance
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    acc_scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    err_scores = 1.0 - acc_scores
    
    cv_results.append({
        'k': k,
        'Mean_Error': round(err_scores.mean(), 5),
        'SD_Error': round(err_scores.std(), 5),
        'Mean_Accuracy': round(acc_scores.mean(), 5),
        'SD_Accuracy': round(acc_scores.std(), 5)
    })

results_df = pd.DataFrame(cv_results)

# Identify optimal k yielding the lowest mean classification error
best_idx = results_df['Mean_Error'].idxmin()
best_k = int(results_df.loc[best_idx, 'k'])
best_err = results_df.loc[best_idx, 'Mean_Error']
best_sd = results_df.loc[best_idx, 'SD_Error']
best_acc = results_df.loc[best_idx, 'Mean_Accuracy']

# Save summary table
table_path = 'results/tables/problem2a_knn_cv_error.csv'
results_df.to_csv(table_path, index=False)
print(f"Results table saved successfully to {table_path}")
print("\nTop 10 values of k with lowest mean classification error:")
print(results_df.sort_values(by=['Mean_Error', 'k']).head(10).to_string(index=False))
print(f"\n>>> Optimal k = {best_k} | Lowest Mean Classification Error = {best_err:.5f} (SD = {best_sd:.5f}) | Mean Accuracy = {best_acc:.5f} <<<\n")

# Plot Problem 2(a): Mean Classification Error vs. k
fig1, ax1 = plt.subplots(figsize=(10.5, 5.5))

ax1.plot(
    results_df['k'], results_df['Mean_Error'],
    marker='o', markersize=5, linewidth=2, color='#1f77b4', label='Mean CV Classification Error'
)
ax1.fill_between(
    results_df['k'],
    results_df['Mean_Error'] - results_df['SD_Error'],
    results_df['Mean_Error'] + results_df['SD_Error'],
    color='#1f77b4', alpha=0.15, label='±1 SD Across 50 Folds'
)

# Highlight optimal k
ax1.axvline(best_k, color='#d62728', linestyle='--', linewidth=1.8, label=f'Optimal k = {best_k} (Error = {best_err:.4f})')
ax1.scatter([best_k], [best_err], color='#d62728', s=95, zorder=5)

ax1.set_title('k-Nearest Neighbors: Classification Error vs. Number of Neighbors (k)\n(Repeated 5-Fold CV, 10 Repeats)', fontsize=12.5, fontweight='bold')
ax1.set_xlabel('Number of Neighbors (k)', fontsize=11)
ax1.set_ylabel('Mean Classification Error (1 - Accuracy)', fontsize=11)
ax1.set_xticks(range(1, 51, 2))
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='best', fontsize=10)

plt.tight_layout()
fig1_path = 'results/figures/problem2a_knn_error_vs_k.png'
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
print(f"Problem 2(a) figure saved successfully to {fig1_path}")
plt.show()

# ---------------------------------------------------------
# Problem 2(b): 2D Decision Boundary (MaxAbsEStateIndex vs. MolWt)
# ---------------------------------------------------------
feat_2d = ['MaxAbsEStateIndex', 'MolWt']
X_2d = df[feat_2d].values
y_2d = y.values

# Fit scaled kNN pipeline using optimal k on the 2D feature space
knn_2d = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k))
knn_2d.fit(X_2d, y_2d)

# Create mesh grid spanning the original physical units of MaxAbsEStateIndex and MolWt
x_pad = (X_2d[:, 0].max() - X_2d[:, 0].min()) * 0.05
y_pad = (X_2d[:, 1].max() - X_2d[:, 1].min()) * 0.05

x_min, x_max = X_2d[:, 0].min() - x_pad, X_2d[:, 0].max() + x_pad
y_min, y_max = X_2d[:, 1].min() - y_pad, X_2d[:, 1].max() + y_pad

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 350), np.linspace(y_min, y_max, 350))
Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig2, ax2 = plt.subplots(figsize=(9.5, 6.5))

# Shaded decision regions (blue = Non-Hydrolase [0], red = Hydrolase EC 3 [1])
ax2.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=['#cfe2f3', '#f4cccc'], alpha=0.75)
ax2.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=1.3, linestyles='--')

# Scatter plot of actual enzyme samples
non_hyd = (y_2d == 0)
hyd = (y_2d == 1)

ax2.scatter(
    X_2d[non_hyd, 0], X_2d[non_hyd, 1],
    c='#1f77b4', marker='o', s=32, alpha=0.75, edgecolors='black', linewidth=0.4,
    label=f'Non-Hydrolase [EC3 = 0] (n={non_hyd.sum()})'
)
ax2.scatter(
    X_2d[hyd, 0], X_2d[hyd, 1],
    c='#d62728', marker='^', s=42, alpha=0.85, edgecolors='black', linewidth=0.45,
    label=f'Hydrolase [EC3 = 1] (n={hyd.sum()})'
)

ax2.set_title(f'k-Nearest Neighbors Decision Boundary (k = {best_k})\nMaxAbsEStateIndex vs. MolWt', fontsize=12.5, fontweight='bold')
ax2.set_xlabel('MaxAbsEStateIndex', fontsize=11)
ax2.set_ylabel('MolWt', fontsize=11)
ax2.legend(loc='upper right', fontsize=10, framealpha=0.92)
ax2.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
fig2_path = 'results/figures/problem2b_knn_decision_boundary.png'
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
print(f"Problem 2(b) figure saved successfully to {fig2_path}")
plt.show()