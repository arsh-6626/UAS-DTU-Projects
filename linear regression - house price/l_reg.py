import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def read_and_split_data(file):
    df = pd.read_csv(file, usecols=[0]) 
    df = df.replace({'yes': 1, 'no': 0, 'furnished': 2, 'unfurnished': 0, 'semi-furnished': 1})
    df = df.sample(frac = 1)
    x_train = df.iloc[0:436, :]
    y_train = np.array(df.iloc[0:436, :])
    x_test = df.iloc[436:, :]
    y_test = np.array(df.iloc[436:, :])
    
    x_train_normalized = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
    x_test_normalized = (x_test - np.mean(x_test, axis=0)) / np.std(x_test, axis=0)
    
    return np.array(x_train_normalized), np.array(x_test_normalized), y_train, y_test

def compute_cost(x, y, w, b):
    m = x.shape[0]
    y_hat = np.dot(x, w) + b
    cost = (1 / (2 * m)) * np.sum((y_hat - y.reshape(-1)) ** 2)
    return cost

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    y_hat = np.dot(x, w) + b
    error = y_hat - y.reshape(-1)
    
    dj_dw = (1/m) * np.dot(x.T, error)
    dj_db = (1/m) * np.sum(error)
    
    return dj_db, dj_dw

def gradient_descent(x, y, learning_rate=0.007, num_iters=10000):
    m, n = x.shape
    w = np.zeros(n)
    b = 0.0
    
    J_history = []
    
    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient(x, y, w, b)
        
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db
        
        current_cost = compute_cost(x, y, w, b)
        J_history.append(current_cost)
        
        if i % 100 == 0:
            print(f"Iteration {i:4d}: Cost {current_cost:8.6f}")
            
    return w, b, J_history

def predict(x, w, b):
    return np.dot(x, w) + b

def calc_metrics(y_true, y_pred):
    ss_res = np.sum((y_true.reshape(-1) - y_pred) ** 2)
    ss_tot = np.sum((y_true.reshape(-1) - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    mse = np.mean((y_true.reshape(-1) - y_pred) ** 2)
    mae = np.mean(np.abs(y_true.reshape(-1) - y_pred))
    
    return mse, mae, r2

def plot_all_graphs(J_history, y_train, y_pred_train, y_test, y_pred_test):
    # Plot cost history
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(J_history)
    plt.title('Cost vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.grid(True)
    
    # Plot predictions vs actual
    plt.subplot(1, 2, 2)
    plt.scatter(y_train, y_pred_train, color='blue', alpha=0.5, label='Training Data')
    plt.scatter(y_test, y_pred_test, color='red', alpha=0.5, label='Test Data')
    
    # Plot perfect prediction line
    all_y = np.concatenate([y_train.reshape(-1), y_test.reshape(-1)])
    plt.plot([all_y.min(), all_y.max()], [all_y.min(), all_y.max()], 'k--', label='Perfect Prediction')
    
    plt.title('Predicted vs Actual Prices')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    x_train, x_test, y_train, y_test = read_and_split_data('Housing.csv')
    
    # Train model
    w, b, J_history = gradient_descent(x_train, y_train, learning_rate=0.001, num_iters=10000)
    
    # Make predictions
    y_pred_train = predict(x_train, w, b)
    y_pred_test = predict(x_test, w, b)
    
    # Calculate metrics
    mse, mae, r2 = calc_metrics(y_test, y_pred_test)
    print("\nTest Set Metrics:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    
    # Plot results
    plot_all_graphs(J_history, y_train, y_pred_train, y_test, y_pred_test)