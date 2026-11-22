import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

# Define the 5 LDA model configurations
models = [
    ('(i) solver="svd"', LinearDiscriminantAnalysis(solver='svd')),
    ('(ii) solver="lsqr", shrinkage=0.0', LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.0)),
    ('(iii) solver="lsqr", shrinkage=0.2', LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.2)),
    ('(iv) solver="lsqr", shrinkage=0.5', LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.5)),
    ('(v) solver="lsqr", shrinkage=1.0', LinearDiscriminantAnalysis(solver='lsqr', shrinkage=1.0)),
]

results = []
fold_scores_dict = {}

for label, clf in models:
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    fold_scores_dict[label] = scores
    results.append({
        'Configuration': label,
        'Solver': 'svd' if 'svd' in label else 'lsqr',
        'Shrinkage': 'None' if 'svd' in label else float(label.split('=')[-1]),
        'Mean_Accuracy': round(scores.mean(), 4),
        'SD_Accuracy': round(scores.std(), 4),
        'Median_Accuracy': round(pd.Series(scores).median(), 4),
        'Min_Accuracy': round(scores.min(), 4),
        'Max_Accuracy': round(scores.max(), 4)
    })

results_df = pd.DataFrame(results)

# Save summary table to results/tables
table_path = 'results/tables/problem1b_lda_accuracies.csv'
results_df.to_csv(table_path, index=False)
print(f"Results table saved successfully to {table_path}\n")
print("LDA Cross-Validation Accuracy Summary (10 repeats x 5 folds = 50 folds):")
print(results_df.to_string(index=False))

# Generate comparison plot and save to results/figures
fig, ax = plt.subplots(figsize=(10, 5))
scores_df = pd.DataFrame(fold_scores_dict)

sns.boxplot(data=scores_df, ax=ax, palette='Blues', width=0.45)
sns.stripplot(data=scores_df, ax=ax, color='black', alpha=0.5, jitter=True, size=5)

# Plot mean accuracy markers
means = [scores_df[col].mean() for col in scores_df.columns]
ax.scatter(range(len(means)), means, color='red', zorder=5, s=60, label='Mean Accuracy')

ax.set_title('LDA Classifier Accuracy Across Solvers and Shrinkage Levels (Repeated 5-Fold CV)', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_xlabel('LDA Configuration', fontsize=11)
ax.set_xticks(range(len(scores_df.columns)))
ax.set_xticklabels(scores_df.columns, rotation=15, ha='right')
ax.legend(loc='lower right')
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
fig_path = 'results/figures/problem1b_lda_accuracy_comparison.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved successfully to {fig_path}")
plt.show()