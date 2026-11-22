import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedKFold, cross_val_score

# Ensure output directories exist
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)

# Load dataset
df = pd.read_csv('data/placental_geneexp.csv')

features = ['LGALS14', 'MGST1', 'ATXN3', 'SSU72', 'ICAM2']
X = df[features]
y = df['SMOKER']

# Repeated k-fold cross-validation: 5 folds, 10 repeats, random_state=10
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=10)

C_values = [0.1, 1, 10]
gamma_values = ['scale', 0.01, 0.1]
kernels = ['linear', 'rbf', 'poly']

results = []

for kernel in kernels:
    if kernel == 'linear':
        # Linear kernel does not use gamma
        for C in C_values:
            clf = SVC(kernel='linear', C=C, random_state=10)
            scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
            results.append({
                'Kernel': 'linear',
                'C': C,
                'Gamma': 'N/A',
                'Mean_ROC_AUC': round(scores.mean(), 4),
                'SD_ROC_AUC': round(scores.std(), 4),
                'Median_ROC_AUC': round(np.median(scores), 4),
                'Min_ROC_AUC': round(scores.min(), 4),
                'Max_ROC_AUC': round(scores.max(), 4)
            })
    else:
        # RBF and Poly (degree=2) kernels use both C and gamma
        for C in C_values:
            for gamma in gamma_values:
                if kernel == 'poly':
                    clf = SVC(kernel='poly', degree=2, C=C, gamma=gamma, random_state=10)
                else:
                    clf = SVC(kernel='rbf', C=C, gamma=gamma, random_state=10)
                scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
                results.append({
                    'Kernel': f'poly (degree=2)' if kernel == 'poly' else 'rbf',
                    'C': C,
                    'Gamma': str(gamma),
                    'Mean_ROC_AUC': round(scores.mean(), 4),
                    'SD_ROC_AUC': round(scores.std(), 4),
                    'Median_ROC_AUC': round(np.median(scores), 4),
                    'Min_ROC_AUC': round(scores.min(), 4),
                    'Max_ROC_AUC': round(scores.max(), 4)
                })

results_df = pd.DataFrame(results)

# Save summary table to results/tables
table_path = 'results/tables/problem1c_svm_roc_auc.csv'
results_df.to_csv(table_path, index=False)
print(f"Results table saved successfully to {table_path}\n")
print("SVM Cross-Validation ROC AUC Summary (10 repeats x 5 folds = 50 folds):")
print(results_df.to_string(index=False))

# Create heatmap plots (one subplot for each kernel)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={'width_ratios': [1.1, 2.2, 2.2]})

# Determine shared color scale across all three heatmaps for fair visual comparison
vmin = results_df['Mean_ROC_AUC'].min()
vmax = results_df['Mean_ROC_AUC'].max()

# 1. Linear Kernel Heatmap (C vs N/A)
linear_df = results_df[results_df['Kernel'] == 'linear'].pivot(index='C', columns='Gamma', values='Mean_ROC_AUC')
linear_df = linear_df.reindex(index=C_values)
sns.heatmap(
    linear_df, annot=True, fmt='.4f', cmap='YlGnBu', cbar=False,
    vmin=vmin, vmax=vmax, ax=axes[0], annot_kws={'size': 11, 'weight': 'bold'}
)
axes[0].set_title('Linear Kernel', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Gamma (Not Applicable)', fontsize=10)
axes[0].set_ylabel('Regularization Parameter (C)', fontsize=11)

# 2. RBF Kernel Heatmap (C vs Gamma)
rbf_df = results_df[results_df['Kernel'] == 'rbf'].pivot(index='C', columns='Gamma', values='Mean_ROC_AUC')
rbf_df = rbf_df.reindex(index=C_values, columns=[str(g) for g in gamma_values])
sns.heatmap(
    rbf_df, annot=True, fmt='.4f', cmap='YlGnBu', cbar=False,
    vmin=vmin, vmax=vmax, ax=axes[1], annot_kws={'size': 11, 'weight': 'bold'}
)
axes[1].set_title('RBF Kernel', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Gamma', fontsize=11)
axes[1].set_ylabel('')

# 3. Polynomial (degree=2) Kernel Heatmap (C vs Gamma)
poly_df = results_df[results_df['Kernel'] == 'poly (degree=2)'].pivot(index='C', columns='Gamma', values='Mean_ROC_AUC')
poly_df = poly_df.reindex(index=C_values, columns=[str(g) for g in gamma_values])
sns.heatmap(
    poly_df, annot=True, fmt='.4f', cmap='YlGnBu', cbar=True,
    cbar_kws={'label': 'Mean ROC AUC'},
    vmin=vmin, vmax=vmax, ax=axes[2], annot_kws={'size': 11, 'weight': 'bold'}
)
axes[2].set_title('Polynomial Kernel (degree=2)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Gamma', fontsize=11)
axes[2].set_ylabel('')

plt.suptitle('SVM Mean ROC AUC Across Kernels and Hyperparameters (Repeated 5-Fold CV)', fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()

fig_path = 'results/figures/problem1c_svm_roc_auc_heatmaps.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved successfully to {fig_path}")
plt.show()