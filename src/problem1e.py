import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold, cross_val_score

# Ensure output directories exist
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)

# Load dataset
df = pd.read_csv('data/placental_geneexp.csv')

features = ['LGALS14', 'MGST1', 'ATXN3', 'SSU72', 'ICAM2']
X = df[features]
y = df['SMOKER']

# Repeated k-fold cross-validation: 5 splits, 10 repeats, random_state=10
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=10)

n_estimators_list = [10, 100]
max_depth_list = [3, 6, None]
min_samples_leaf_list = [1, 5, 15]

results = []

for n_est in n_estimators_list:
    for depth in max_depth_list:
        for min_leaf in min_samples_leaf_list:
            clf = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=depth,
                min_samples_leaf=min_leaf,
                random_state=10
            )
            scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
            results.append({
                'n_estimators': n_est,
                'max_depth': 'None' if depth is None else str(depth),
                'min_samples_leaf': min_leaf,
                'Mean_ROC_AUC': round(scores.mean(), 4),
                'SD_ROC_AUC': round(scores.std(), 4),
                'Median_ROC_AUC': round(np.median(scores), 4),
                'Min_ROC_AUC': round(scores.min(), 4),
                'Max_ROC_AUC': round(scores.max(), 4)
            })

results_df = pd.DataFrame(results)

# Save summary table to results/tables
table_path = 'results/tables/problem1e_rf_roc_auc.csv'
results_df.to_csv(table_path, index=False)
print(f"Results table saved successfully to {table_path}\n")
print("Random Forest Cross-Validation ROC AUC Summary (10 repeats x 5 folds = 50 folds):")
print(results_df.to_string(index=False))

# Create figure with 2 subplots (n_estimators = 10 and 100)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

depth_order = ['3', '6', 'None']
leaf_order = [1, 5, 15]
palette = ['#2b5c8f', '#4e9af1', '#a6cee3']

for idx, n_est in enumerate(n_estimators_list):
    ax = axes[idx]
    sub_df = results_df[results_df['n_estimators'] == n_est]
    
    sns.barplot(
        data=sub_df,
        x='max_depth',
        y='Mean_ROC_AUC',
        hue='min_samples_leaf',
        order=depth_order,
        hue_order=leaf_order,
        palette=palette,
        edgecolor='black',
        linewidth=0.8,
        ax=ax
    )
    
    # Add exact value labels on top of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8.5, fontweight='bold')
        
    ax.set_title(f'Random Forest ({n_est} Decision Trees)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Maximum Tree Depth (max_depth)', fontsize=11)
    if idx == 0:
        ax.set_ylabel('Mean Cross-Validation ROC AUC', fontsize=11)
    else:
        ax.set_ylabel('')
        
    ax.set_ylim(0.5, 0.95)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(title='min_samples_leaf', loc='lower right', fontsize=9.5, title_fontsize=10)

plt.suptitle('Random Forest Mean ROC AUC Across Hyperparameters (Repeated 5-Fold CV)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

fig_path = 'results/figures/problem1e_rf_roc_auc_bars.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved successfully to {fig_path}")
plt.show()