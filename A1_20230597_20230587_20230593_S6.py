import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data / 255.0
y = mnist.target.astype(int)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=10000, test_size=2000, stratify=y, random_state=0
)


print("\nTraining Gaussian Models...")
classes = np.unique(y_train)
means = {}
covs = {}
priors = {}

for c in classes:
    Xc = X_train[y_train == c]
    means[c] = Xc.mean(axis=0)
    covs[c] = np.cov(Xc, rowvar=False) + 1e-6 * np.eye(Xc.shape[1])
    priors[c] = len(Xc) / len(X_train)

def gaussian_log_likelihood(X, mean, cov):
    try:
        return multivariate_normal.logpdf(X, mean=mean, cov=cov)
    except Exception as e:
        try:
            d = X.shape[1]
            cov_reg = cov + 1e-6 * np.eye(d)
            return multivariate_normal.logpdf(X, mean=mean, cov=cov_reg)
        except Exception:
            d = X.shape[1]
            diff = X - mean
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                cov = cov + 1e-6 * np.eye(d)
                sign, logdet = np.linalg.slogdet(cov)
            cov_inv = np.linalg.inv(cov)
            exp_term = np.einsum('ij,jk,ik->i', diff, cov_inv, diff)
            log_norm_const = 0.5 * (d * np.log(2 * np.pi) + logdet)
            return -0.5 * exp_term - log_norm_const

log_probs_gauss = np.zeros((X_test.shape[0], len(classes)))
for i, c in enumerate(classes):
    log_probs_gauss[:, i] = gaussian_log_likelihood(X_test, means[c], covs[c]) + np.log(priors[c])

y_pred_gauss = np.argmax(log_probs_gauss, axis=1)
acc_gauss = accuracy_score(y_test, y_pred_gauss)
print(f"Gaussian Model Accuracy: {acc_gauss:.4f}")


print("\nTraining Gaussian Mixture Models (K=3)...")
K = 3
gmms = {}
for c in classes:
    Xc = X_train[y_train == c]
    gmms[c] = GaussianMixture(n_components=K, covariance_type='full', max_iter=50, random_state=0)
    gmms[c].fit(Xc)
log_probs_gmm = np.zeros((X_test.shape[0], len(classes)))
for i, c in enumerate(classes):
    log_probs_gmm[:, i] = gmms[c].score_samples(X_test) + np.log(priors[c])

y_pred_gmm = np.argmax(log_probs_gmm, axis=1)
acc_gmm = accuracy_score(y_test, y_pred_gmm)
print(f"GMM Model Accuracy: {acc_gmm:.4f}")

y_test_bin = label_binarize(y_test, classes=classes)

fpr_gauss, tpr_gauss, auc_gauss = {}, {}, {}
for i, c in enumerate(classes):
    fpr_gauss[c], tpr_gauss[c], _ = roc_curve(y_test_bin[:, i], log_probs_gauss[:, i])
    auc_gauss[c] = auc(fpr_gauss[c], tpr_gauss[c])


fpr_gmm, tpr_gmm, auc_gmm = {}, {}, {}
for i, c in enumerate(classes):
    fpr_gmm[c], tpr_gmm[c], _ = roc_curve(y_test_bin[:, i], log_probs_gmm[:, i])
    auc_gmm[c] = auc(fpr_gmm[c], tpr_gmm[c])

n_classes = len(classes)
cols = 5
rows = int(np.ceil(n_classes / cols))
fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True, sharey=True)
axes = axes.flatten()

for idx, c in enumerate(classes):
    ax = axes[idx]
    ax.plot(fpr_gauss[c], tpr_gauss[c], linestyle='--', color='tab:blue', label=f'Gaussian (AUC={auc_gauss[c]:.2f})')
    ax.plot(fpr_gmm[c], tpr_gmm[c], linestyle='-', color='tab:red', label=f'GMM (AUC={auc_gmm[c]:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=0.8)
    ax.set_title(f'Class {c}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(loc='lower right', fontsize='small')

for j in range(n_classes, rows * cols):
    fig.delaxes(axes[j])

fig.suptitle("ROC Curves for MNIST (Gaussian vs GMM) — per-class comparison", fontsize=16)
left_margin = 0.12
bottom_margin = 0.06
try:
    fig.supxlabel('False Positive Rate', fontsize=12, y=bottom_margin / 2)
    fig.supylabel('True Positive Rate', fontsize=12, x=left_margin / 2)
except Exception:
    fig.text(0.5, bottom_margin / 2, 'False Positive Rate', ha='center', va='center')
    fig.text(left_margin / 2, 0.5, 'True Positive Rate', va='center', ha='center', rotation='vertical')

plt.tight_layout(rect=[left_margin, bottom_margin, 1, 0.96])

print("\n===== Comparative Results =====", flush=True)
print(f"Gaussian Accuracy: {acc_gauss:.4f}", flush=True)
print(f"GMM Accuracy:      {acc_gmm:.4f}", flush=True)
print("\nAverage AUC per class:", flush=True)
print(f"Gaussian mean AUC: {np.mean(list(auc_gauss.values())):.4f}", flush=True)
print(f"GMM mean AUC:      {np.mean(list(auc_gmm.values())):.4f}", flush=True)

plt.show()