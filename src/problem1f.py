import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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

# Define the 4 classifiers with default hyperparameters
classifiers = [
    ('Quadratic Discriminant Analysis (QDA)', QuadraticDiscriminantAnalysis()),
    ('SVM (RBF Kernel)', SVC(kernel='rbf', random_state=10)),
    ('Gaussian Naïve Bayes', GaussianNB()),
    ('Random Forest (100 Trees)', RandomForestClassifier(n_estimators=100, random_state=10))
]

summary_rows = []
avg_cms = {}

for name, clf in classifiers:
    cm_list = []
    acc_list, prec_list, rec_list, f1_list = [], [], [], []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cm_list.append(cm)
        
        acc_list.append(accuracy_score(y_test, y_pred))
        prec_list.append(precision_score(y_test, y_pred, zero_division=0))
        rec_list.append(recall_score(y_test, y_pred, zero_division=0))
        f1_list.append(f1_score(y_test, y_pred, zero_division=0))
    
    # Average confusion matrix across all 50 folds (10 repeats x 5 folds)
    avg_cm = np.mean(cm_list, axis=0)
    avg_cms[name] = avg_cm
    
    tn, fp, fn, tp = avg_cm.ravel()
    summary_rows.append({
        'Classifier': name,
        'Avg_TN': round(tn, 2),
        'Avg_FP': round(fp, 2),
        'Avg_FN': round(fn, 2),
        'Avg_TP': round(tp, 2),
        'Mean_Accuracy': round(np.mean(acc_list), 4),
        'SD_Accuracy': round(np.std(acc_list), 4),
        'Mean_Precision': round(np.mean(prec_list), 4),
        'Mean_Recall': round(np.mean(rec_list), 4),
        'Mean_F1': round(np.mean(f1_list), 4)
    })

summary_df = pd.DataFrame(summary_rows)

# Save summary table to results/tables
table_path = 'results/tables/problem1f_classifier_comparison.csv'
summary_df.to_csv(table_path, index=False)
print(f"Summary table saved successfully to {table_path}\n")
print("Average Confusion Matrix & Performance Across 50 Folds (10 Repeats x 5 Folds):")
print(summary_df.to_string(index=False))

# ---------------------------------------------------------
# Figure 1: Average Confusion Matrices (2x2 Grid)
# ---------------------------------------------------------
fig1, axes1 = plt.subplots(2, 2, figsize=(11, 9))
axes1 = axes1.ravel()

class_labels = ['Non-Smoker (0)', 'Smoker (1)']

for idx, (name, avg_cm) in enumerate(avg_cms.items()):
    ax = axes1[idx]
    # Compute row percentages alongside average fold counts
    row_sums = avg_cm.sum(axis=1, keepdims=True)
    cm_pct = (avg_cm / row_sums) * 100.0
    
    annot = np.empty_like(avg_cm, dtype=object)
    for r in range(2):
        for c in range(2):
            annot[r, c] = f"{avg_cm[r, c]:.2f}\n({cm_pct[r, c]:.1f}%)"
            
    sns.heatmap(
        avg_cm, annot=annot, fmt='', cmap='Blues', cbar=False,
        xticklabels=class_labels, yticklabels=class_labels,
        vmin=0, vmax=7.2, ax=ax,
        annot_kws={'size': 11, 'weight': 'bold'},
        linewidths=1, linecolor='black'
    )
    ax.set_title(name, fontsize=11.5, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=10.5)
    ax.set_ylabel('True Label', fontsize=10.5)

plt.suptitle('Average Confusion Matrix per Fold (Repeated 5-Fold CV, 10 Repeats)', fontsize=13.5, fontweight='bold', y=1.01)
plt.tight_layout()

cm_fig_path = 'results/figures/problem1f_avg_confusion_matrices.png'
plt.savefig(cm_fig_path, dpi=300, bbox_inches='tight')
print(f"\nConfusion matrix figure saved successfully to {cm_fig_path}")
plt.show()

# ---------------------------------------------------------
# Figure 2: 2D Decision Boundaries on LGALS14 vs ICAM2
# ---------------------------------------------------------
X_2d = df[['LGALS14', 'ICAM2']].values
y_2d = df['SMOKER'].values

x_min, x_max = X_2d[:, 0].min() - 0.35, X_2d[:, 0].max() + 0.35
y_min, y_max = X_2d[:, 1].min() - 0.35, X_2d[:, 1].max() + 0.35
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Re-instantiate fresh default classifiers for 2D visualization
classifiers_2d = [
    ('Quadratic Discriminant Analysis (QDA)', QuadraticDiscriminantAnalysis()),
    ('SVM (RBF Kernel)', SVC(kernel='rbf', random_state=10)),
    ('Gaussian Naïve Bayes', GaussianNB()),
    ('Random Forest (100 Trees)', RandomForestClassifier(n_estimators=100, random_state=10))
]

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
axes2 = axes2.ravel()

for idx, (name, clf_2d) in enumerate(classifiers_2d):
    ax = axes2[idx]
    clf_2d.fit(X_2d, y_2d)
    
    Z = clf_2d.predict(grid_points).reshape(xx.shape)
    
    # Shaded decision regions (blue = Non-Smoker 0, red = Smoker 1)
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=['#cfe2f3', '#f4cccc'], alpha=0.75)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=1.5, linestyles='--')
    
    # Scatter plot of actual samples
    non_smokers = (y_2d == 0)
    smokers = (y_2d == 1)
    ax.scatter(
        X_2d[non_smokers, 0], X_2d[non_smokers, 1],
        c='#1f77b4', marker='o', s=65, edgecolors='black', linewidth=0.8, label='Non-Smoker (0)'
    )
    ax.scatter(
        X_2d[smokers, 0], X_2d[smokers, 1],
        c='#d62728', marker='^', s=75, edgecolors='black', linewidth=0.8, label='Smoker (1)'
    )
    
    ax.set_title(name, fontsize=11.5, fontweight='bold')
    if idx in [2, 3]:
        ax.set_xlabel('LGALS14 Expression', fontsize=11)
    if idx in [0, 2]:
        ax.set_ylabel('ICAM2 Expression', fontsize=11)
    ax.legend(loc='upper right', fontsize=9.5, framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.5)

plt.suptitle('Decision Boundaries on LGALS14 vs. ICAM2 (Default Hyperparameters)', fontsize=13.5, fontweight='bold', y=1.01)
plt.tight_layout()

db_fig_path = 'results/figures/problem1f_decision_boundaries_2d.png'
plt.savefig(db_fig_path, dpi=300, bbox_inches='tight')
print(f"Decision boundary figure saved successfully to {db_fig_path}")
plt.show()