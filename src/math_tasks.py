import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
import os

def load_data(filepath):
    """Sử dụng pandas để đọc file cho chính xác tên cột, sau đó xử lý NaN và chuyển sang NumPy."""
    df = pd.read_csv(filepath)
    df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
    return df

def q1_matrix_representation(df):
    print("--- Q1: Matrix Representation ---")
    X = df[['age', 'avg_glucose_level', 'bmi']].values
    shape = X.shape
    rank = np.linalg.matrix_rank(X)
    print(f"Matrix Shape: {shape}")
    print(f"Matrix Rank: {rank}\n")
    return X

def q2_linear_regression(df):
    print("--- Q2: Linear Regression (Normal Equation) ---")
    X = df[['age', 'avg_glucose_level']].values
    y = df['stroke'].values
    X_b = np.c_[np.ones((X.shape[0], 1)), X]

    w = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    print(f"Weights (b, w1, w2): {w}\n")
    return X_b, y, w

def q3_probabilities(df):
    print("--- Q3: Base and Conditional Probabilities ---")
    p_stroke = df['stroke'].mean()
    
    hyper_df = df[df['hypertension'] == 1]
    p_stroke_given_hyper = hyper_df['stroke'].mean() if not hyper_df.empty else 0
    
    print(f"P(stroke=1): {p_stroke:.4f}")
    print(f"P(stroke=1 | hypertension=1): {p_stroke_given_hyper:.4f}\n")

def q4_gradient_mse(X_b, y, w):
    print("--- Q4: Compute Gradient of MSE ---")
    n = len(y)
    y_pred = X_b.dot(w)
    gradient = -(2/n) * X_b.T.dot(y - y_pred)
    print(f"Gradient with respect to W: {gradient}\n")

def q5_variance_covariance(df):
    print("--- Q5: Variance and Covariance ---")
    glucose = df['avg_glucose_level'].values
    age = df['age'].values
    
    var_glucose = np.var(glucose)
    cov_matrix = np.cov(glucose, age)
    cov_glucose_age = cov_matrix[0, 1]
    
    print(f"Variance of avg_glucose_level: {var_glucose:.2f}")
    print(f"Covariance between glucose and age: {cov_glucose_age:.2f}\n")

def q6_covariance_eigen(X):
    print("--- Q6: Covariance Matrix & Eigen Decomposition ---")
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    cov_matrix = np.cov(X_std, rowvar=False)
    
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    print("Covariance Matrix:\n", cov_matrix)
    print("Eigenvalues:\n", eigenvalues)
    print("Eigenvectors:\n", eigenvectors, "\n")

def q7_svd(X):
    print("--- Q7: Singular Value Decomposition (SVD) ---")
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
    
    print(f"Sigma (Singular values): {S}")
    print(f"V^T (Principal components): \n{Vt}\n")

def q8_gradient_descent(X_b, y):
    print("--- Q8: Gradient Descent & Loss Visualization ---")
    X_b_scaled = np.copy(X_b)
    X_b_scaled[:, 1:] = (X_b[:, 1:] - np.mean(X_b[:, 1:], axis=0)) / np.std(X_b[:, 1:], axis=0)
    
    n = len(y)
    lr = 0.01
    epochs = 200
    w = np.zeros(X_b_scaled.shape[1])
    losses = []
    
    for _ in range(epochs):
        y_pred = X_b_scaled.dot(w)
        loss = (1/n) * np.sum((y - y_pred)**2)
        losses.append(loss)
        grad = -(2/n) * X_b_scaled.T.dot(y - y_pred)
        w -= lr * grad
        
    print(f"Final Weights after {epochs} epochs: {w}")
    print(f"Final Loss: {losses[-1]:.4f}\n")

    os.makedirs('outputs/figures', exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), losses, color='blue', linewidth=2)
    plt.title('Loss Convergence over Iterations')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True)
    plt.savefig('outputs/figures/loss_convergence.png')
    print("-> Saved plot to 'outputs/figures/loss_convergence.png'\n")

def q9_bayes_theorem(df):
    print("--- Q9: Bayes' Theorem with Discretization ---")
    bins = [0, 100, 125, float('inf')]
    labels = ['Normal', 'Prediabetic', 'Diabetic']
    df['glucose_bin'] = pd.cut(df['avg_glucose_level'], bins=bins, labels=labels)
    
    p_stroke_given_bin = df.groupby('glucose_bin', observed=False)['stroke'].mean()
    print("P(stroke | glucose_bin):")
    print(p_stroke_given_bin, "\n")

def q10_regularization(X_b, y):
    print("--- Q10: L1 (Lasso) vs L2 (Ridge) Regularization ---")
    X_features = X_b[:, 1:] 
    
    lasso = Lasso(alpha=0.1).fit(X_features, y)
    ridge = Ridge(alpha=0.1).fit(X_features, y)
    
    print(f"Lasso (L1) Coefficients: {lasso.coef_}")
    print(f"Ridge (L2) Coefficients: {ridge.coef_}\n")

if __name__ == "__main__":
    filepath = 'data/raw/healthcare-dataset-stroke-data.csv'
    df = load_data(filepath)
    
    X_num = q1_matrix_representation(df)
    X_b, y, w_optimal = q2_linear_regression(df)
    q3_probabilities(df)
    q4_gradient_mse(X_b, y, w_optimal)
    q5_variance_covariance(df)
    q6_covariance_eigen(X_num)
    q7_svd(X_num)
    q8_gradient_descent(X_b, y)
    q9_bayes_theorem(df)
    q10_regularization(X_b, y)
