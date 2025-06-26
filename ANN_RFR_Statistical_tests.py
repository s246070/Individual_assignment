#%%
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv(r"C:\Users\45237\OneDrive - Danmarks Tekniske Universitet\Skrivebord\assignment\assignment\task_2\HR_data.csv")  # Adjust path if needed

# Define features and target
features = ['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min', 'HR_Max', 'HR_AUC']
X = df[features].values
y = df['Frustrated'].values.reshape(-1, 1)
groups = df['Individual'].values

#%%
# Settings
n_splits = 14
epochs = 200
lr = 0.01
hidden_units = 8

# Create and store GroupKFold splits
gkf = GroupKFold(n_splits=n_splits)
ann_mae_scores = []
all_learning_curves = []  # store per-fold loss curves

#%%
# Define model
def create_model(input_dim, hidden_units):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_units),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_units, 1)
    )

# GroupK Cross-validation
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
    print(f"\nFold {fold+1}/{n_splits}")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float)

    # Model and optimizer
    model = create_model(X_train.shape[1], hidden_units)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # Training loop with loss tracking
    fold_curve = []
    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train_tensor)
        loss = loss_fn(y_pred, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        fold_curve.append(loss.item())

    all_learning_curves.append(fold_curve)

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_tensor).numpy()
        mae = mean_absolute_error(y_test, y_pred_test)
        ann_mae_scores.append(mae)
        print(f"MAE: {mae:.3f}")



print("\nCross-Validation MAE Scores:", ann_mae_scores)
print("Mean MAE:", np.mean(ann_mae_scores))
print(f"Standard Deviation of MAE: {np.std(ann_mae_scores):.3f}")

#%% Plot learning curves
plt.figure(figsize=(10, 6))
for fold_idx, curve in enumerate(all_learning_curves):
    plt.plot(curve, label=f'Fold {fold_idx + 1}', alpha=0.7)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Learning Curves for ANN over CV Folds")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()


# RF
#%% Random Forest Regression with GroupKFold
from sklearn.ensemble import RandomForestRegressor

rf_mae_scores = []
L = 400 # number of rounds of bagging

# GroupK Cross-validation
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
    print(f"\n[RF] Fold {fold+1}/{n_splits}")

    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Standardize (optional for tree models — you can skip if preferred)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model
    rf = RandomForestRegressor(n_estimators=L, random_state=fold)
    rf.fit(X_train, y_train.ravel())  # flatten target for sklearn

    # Prediction and evaluation
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rf_mae_scores.append(mae)
    print(f"[RF] MAE: {mae:.3f}")

# Summary
print("\n[RF] Cross-Validation MAE Scores:", rf_mae_scores)
print("[RF] Mean MAE:", np.mean(rf_mae_scores))
print(f"[RF] Standard Deviation of MAE: {np.std(rf_mae_scores):.3f}")


#%%
# Shapiro wilks test
from scipy.stats import shapiro

print("ANN Normality p-value:", shapiro(ann_mae_scores).pvalue)
print("RF Normality p-value:", shapiro(rf_mae_scores).pvalue)

#%%
print(len(ann_mae_scores),len(rf_mae_scores))
#%%
from scipy.stats import ttest_rel

# Perform paired t-test
t_stat, p_value = ttest_rel(ann_mae_scores, rf_mae_scores)

print(f"Paired t-test t-statistic: {t_stat:.3f}")
print(f"Paired t-test p-value: {p_value:.4f}")
