import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------
# 1. Load MNIST
# -----------------------------
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data / 255.0
y = mnist.target.astype(int)

# Use a subset for speed (optional)
n_train = 10000
n_test = 2000
X_train, X_test = X[:n_train], X[-n_test:]
y_train, y_test = y[:n_train], y[-n_test:]

# -----------------------------
# 2. Gaussian Model per Class
# -----------------------------
print("\nTraining Gaussian Models...")
classes = np.unique(y_train)
means = {}
covs = {}
priors = {}

for c in classes:
    Xc = X_train[y_train == c]
    means[c] = Xc.mean(axis=0)
    covs[c] = np.cov(Xc, rowvar=False) + 1e-6 * np.eye(Xc.shape[1])  # Regularize
    priors[c] = len(Xc) / len(X_train)

def gaussian_log_likelihood(X, mean, cov):
    """Compute the log of the multivariate Gaussian PDF for each row in X.

    Returns an array of shape (n_samples,) with log p(x|mean,cov).
    Computation is done in the log-domain using slogdet for numerical stability.
    """
    d = X.shape[1]
    diff = X - mean
    # Use slogdet for a stable log-determinant
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        # Fall back to small regularization if covariance is not positive definite
        cov = cov + 1e-6 * np.eye(d)
        sign, logdet = np.linalg.slogdet(cov)
    cov_inv = np.linalg.inv(cov)
    exp_term = np.einsum('ij,jk,ik->i', diff, cov_inv, diff)
    log_norm_const = 0.5 * (d * np.log(2 * np.pi) + logdet)
    return -0.5 * exp_term - log_norm_const

# Compute log-likelihoods for test data
log_probs_gauss = np.zeros((X_test.shape[0], len(classes)))
for i, c in enumerate(classes):
    # gaussian_log_likelihood returns log p(x|c); add log prior for class posterior (up to constant)
    log_probs_gauss[:, i] = gaussian_log_likelihood(X_test, means[c], covs[c]) + np.log(priors[c])

y_pred_gauss = np.argmax(log_probs_gauss, axis=1)
acc_gauss = accuracy_score(y_test, y_pred_gauss)
print(f"Gaussian Model Accuracy: {acc_gauss:.4f}")

# -----------------------------
# 3. Gaussian Mixture Model per Class
# -----------------------------
print("\nTraining Gaussian Mixture Models (K=3)...")
K = 3
gmms = {}
for c in tqdm(classes):
    Xc = X_train[y_train == c]
    gmms[c] = GaussianMixture(n_components=K, covariance_type='full', max_iter=50, random_state=0)
    gmms[c].fit(Xc)

log_probs_gmm = np.zeros((X_test.shape[0], len(classes)))
for i, c in enumerate(classes):
    log_probs_gmm[:, i] = gmms[c].score_samples(X_test) + np.log(priors[c])

y_pred_gmm = np.argmax(log_probs_gmm, axis=1)
acc_gmm = accuracy_score(y_test, y_pred_gmm)
print(f"GMM Model Accuracy: {acc_gmm:.4f}")

# -----------------------------
# 4. ROC Curves
# -----------------------------
y_test_bin = label_binarize(y_test, classes=classes)

# Gaussian ROC
fpr_gauss, tpr_gauss, auc_gauss = {}, {}, {}
for i, c in enumerate(classes):
    fpr_gauss[c], tpr_gauss[c], _ = roc_curve(y_test_bin[:, i], log_probs_gauss[:, i])
    auc_gauss[c] = auc(fpr_gauss[c], tpr_gauss[c])

# GMM ROC
fpr_gmm, tpr_gmm, auc_gmm = {}, {}, {}
for i, c in enumerate(classes):
    fpr_gmm[c], tpr_gmm[c], _ = roc_curve(y_test_bin[:, i], log_probs_gmm[:, i])
    auc_gmm[c] = auc(fpr_gmm[c], tpr_gmm[c])

# Plot ROCs
plt.figure(figsize=(14, 10))
for c in classes:
    plt.plot(fpr_gauss[c], tpr_gauss[c], linestyle='--', label=f'Gaussian (class {c}, AUC={auc_gauss[c]:.2f})')
    plt.plot(fpr_gmm[c], tpr_gmm[c], label=f'GMM (class {c}, AUC={auc_gmm[c]:.2f})')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for MNIST (Gaussian vs GMM)")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 5. Summary Report
# -----------------------------
print("\n===== Comparative Results =====")
print(f"Gaussian Accuracy: {acc_gauss:.4f}")
print(f"GMM Accuracy:      {acc_gmm:.4f}")
print("\nAverage AUC per class:")
print(f"Gaussian mean AUC: {np.mean(list(auc_gauss.values())):.4f}")
print(f"GMM mean AUC:      {np.mean(list(auc_gmm.values())):.4f}")