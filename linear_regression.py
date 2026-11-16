import numpy as np
import matplotlib.pyplot as plt

# fixed random seed for reproducibility
np.random.seed(42)

# define the number of samples and features
n = 100  # number of samples
m = 3    # number of features

# 1.generate a random design matrix X (n x m) with values from a uniform distribution from 0 to 1
X = np.random.rand(n, m)

# 2. define the true coefficients beta (m x 1)
beta = np.array([[2.5], 
                 [-4.0], 
                 [1.2]])

# 3. generate the noise vector (standard normal distribution, sigma=1)
eps = np.random.randn(n, 1)

# 4. calculate the target vector Y
Y = X @ beta + eps

# 5. solve the normal equations to find beta_hat
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ Y

# 6. compare the true and estimated coefficients
error = np.linalg.norm(beta - beta_hat)

print("True beta:\n", beta.flatten())
print("Estimated beta_hat:\n", np.round(beta_hat.flatten(), 4))
print(f"L2 Estimation Error: {error:.4f}")  

# 7. Experiment: Increase noise standard deviation (sigma) and evaluate its effect
sigmas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
num_trials = 50  # Run multiple Monte Carlo trials per sigma for statistical stability

mean_errors = []
std_errors = []

for sigma in sigmas:
    trial_errors = []
    for _ in range(num_trials):
        # Scale standard normal noise by sigma to obtain N(0, sigma^2)
        eps_sigma = sigma * np.random.randn(n, 1)
        Y_sigma = X @ beta + eps_sigma
        
        # Estimate parameters via OLS closed-form solution
        beta_hat_sigma = np.linalg.inv(X.T @ X) @ X.T @ Y_sigma
        
        # Record Euclidean (L2) estimation error
        err = np.linalg.norm(beta - beta_hat_sigma)
        trial_errors.append(err)
        
    mean_errors.append(np.mean(trial_errors))
    std_errors.append(np.std(trial_errors))

# Generate summary visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left subplot: L2 parameter estimation error as a function of noise level (sigma)
axes[0].errorbar(sigmas, mean_errors, yerr=std_errors, fmt='-o', capsize=5, color='navy', 
                 label='Mean L2 Error ± 1 STD')
axes[0].set_title('Parameter Estimation Error vs. Noise Level (σ)')
axes[0].set_xlabel('Noise Standard Deviation (σ)')
axes[0].set_ylabel(r'Estimation Error $\|\beta - \hat{\beta}\|_2$')
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend()

# Right subplot: 1D visual intuition (m = 1) comparing low vs. high noise regimes
x_1d = np.random.rand(50, 1)
true_slope = 3.0
x_line = np.linspace(0, 1, 100).reshape(-1, 1)

for s, color, label in [(0.2, 'green', 'Low Noise (σ=0.2)'), (2.5, 'crimson', 'High Noise (σ=2.5)')]:
    y_1d = x_1d * true_slope + s * np.random.randn(50, 1)
    slope_hat = (np.linalg.inv(x_1d.T @ x_1d) @ x_1d.T @ y_1d)[0, 0]
    
    axes[1].scatter(x_1d, y_1d, color=color, alpha=0.5, label=f'Data ({label})')
    axes[1].plot(x_line, x_line * slope_hat, color=color, linestyle='--', linewidth=2, 
                 label=f'Estimated (β̂={slope_hat:.2f})')

axes[1].plot(x_line, x_line * true_slope, color='black', linewidth=2.5, label=f'True Model (β={true_slope})')
axes[1].set_title('1D Visualization: Effect of σ on Fitted Line')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend()

plt.tight_layout()
plt.savefig('sigma_experiment_results.png', dpi=300)
plt.show()