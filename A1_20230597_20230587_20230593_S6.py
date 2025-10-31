import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def _logsumexp(a, axis=1, keepdims=False):
    a = np.asarray(a)
    max_a = np.max(a, axis=axis, keepdims=True)
    res = max_a + np.log(np.sum(np.exp(a - max_a), axis=axis, keepdims=True))
    if not keepdims:
        return res.squeeze(axis=axis)
    return res

def gaussian_logpdf_cholesky(X, mean, cov):
    X = np.asarray(X)
    mean = np.asarray(mean)
    d = X.shape[1]
    diff = X - mean
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = cov + 1e-6 * np.eye(d)
        L = np.linalg.cholesky(cov)
    y = np.linalg.solve(L, diff.T)
    quad = np.sum(y * y, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    log_norm_const = 0.5 * (d * np.log(2 * np.pi) + logdet)
    logpdf = -0.5 * quad - log_norm_const
    return logpdf

def initialize_gmm(X, K, rng):
    n_samples, n_features = X.shape
    idx = rng.choice(n_samples, size=K, replace=False)
    means = X[idx].copy()
    cov_common = np.cov(X, rowvar=False) + 1e-6 * np.eye(n_features)
    covs = np.array([cov_common.copy() for _ in range(K)])
    weights = np.ones(K) / K
    return means, covs, weights

def fit_gmm_em(X, K=7, max_iter=100, tol=1e-4, verbose=False, seed=0):
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape
    means, covs, weights = initialize_gmm(X, K, rng)

    prev_ll = -np.inf
    for it in range(1, max_iter + 1):
        log_comp = np.zeros((n_samples, K))
        for k in range(K):
            log_comp[:, k] = gaussian_logpdf_cholesky(X, means[k], covs[k]) + np.log(weights[k] + 1e-12)

        log_prob_sum = _logsumexp(log_comp, axis=1, keepdims=True)
        log_resp = log_comp - log_prob_sum
        responsibilities = np.exp(log_resp)

        Nk = responsibilities.sum(axis=0)
        for k in range(K):
            r_k = responsibilities[:, k]
            means[k] = (r_k[:, None] * X).sum(axis=0) / (Nk[k] + 1e-12)
            diff = X - means[k]
            cov_k = (r_k[:, None, None] * np.einsum('ij,ik->ijk', diff, diff)).sum(axis=0) / (Nk[k] + 1e-12)
            covs[k] = cov_k + 1e-6 * np.eye(n_features)
            weights[k] = Nk[k] / n_samples

        new_ll = np.sum(log_prob_sum)
        if verbose:
            print(f"  EM iter {it:3d}: log-likelihood = {new_ll:.6f} (Δ={new_ll - prev_ll:+.6e})")
        if np.isfinite(new_ll) and abs(new_ll - prev_ll) < tol:
            if verbose:
                print(f"  Converged at iter {it} (Δll={abs(new_ll - prev_ll):.2e})")
            break
        prev_ll = new_ll

    return means, covs, weights

def gmm_log_likelihood(X, means, covs, weights):
    n_samples = X.shape[0]
    K = len(weights)
    log_comp = np.zeros((n_samples, K))
    for k in range(K):
        log_comp[:, k] = gaussian_logpdf_cholesky(X, means[k], covs[k]) + np.log(weights[k] + 1e-12)
    return _logsumexp(log_comp, axis=1)

def compute_roc_auc(y_true_bin, scores):
    thresholds = np.sort(np.unique(scores))[::-1]
    tpr_list, fpr_list = [], []
    P = np.sum(y_true_bin == 1)
    N = np.sum(y_true_bin == 0)

    for thresh in thresholds:
        y_pred = (scores >= thresh).astype(int)
        TP = np.sum((y_pred == 1) & (y_true_bin == 1))
        FP = np.sum((y_pred == 1) & (y_true_bin == 0))
        TPR = TP / P if P > 0 else 0.0
        FPR = FP / N if N > 0 else 0.0
        tpr_list.append(TPR)
        fpr_list.append(FPR)

    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)

    if fpr.size == 0 or fpr[0] != 0.0:
        fpr = np.concatenate(([0.0], fpr))
        tpr = np.concatenate(([0.0], tpr))

    if fpr[-1] != 1.0:
        fpr = np.concatenate((fpr, [1.0]))
        tpr = np.concatenate((tpr, [1.0]))

    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]

    auc_value = np.trapezoid(tpr, fpr)
    return fpr, tpr, auc_value

def main(random_seed=0,
         pca_components=40,
         train_size=10000,
         test_size=2000,
         gmm_K=7,
         gmm_max_iter=80,
         verbose_gmm=False):
    rng = np.random.default_rng(random_seed)

    print("Loading MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data.astype(np.float64) / 255.0
    y = mnist.target.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size, stratify=y, random_state=random_seed
    )

    print(f"Training PCA (n_components={pca_components}) on training set...")
    pca = PCA(n_components=pca_components, random_state=random_seed)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    classes = np.unique(y_train)
    n_classes = len(classes)
    priors = {}
    gauss_means = {}
    gauss_covs = {}

    print("\nFitting single Gaussian per class...")
    for c in classes:
        Xc = X_train_pca[y_train == c]
        gauss_means[c] = Xc.mean(axis=0)
        gauss_covs[c] = np.cov(Xc, rowvar=False) + 1e-6 * np.eye(Xc.shape[1])
        priors[c] = Xc.shape[0] / X_train_pca.shape[0]

    log_probs_gauss = np.zeros((X_test_pca.shape[0], n_classes))
    for i, c in enumerate(classes):
        log_probs_gauss[:, i] = gaussian_logpdf_cholesky(X_test_pca, gauss_means[c], gauss_covs[c]) + np.log(priors[c])

    y_pred_gauss = np.argmax(log_probs_gauss, axis=1)
    acc_gauss = accuracy_score(y_test, y_pred_gauss)
    print(f"Gaussian Model Accuracy: {acc_gauss:.4f}")

    gmm_params = {}
    for c in classes:
        Xc = X_train_pca[y_train == c]
        means_c, covs_c, weights_c = fit_gmm_em(Xc, K=gmm_K, max_iter=gmm_max_iter, tol=1e-4, verbose=verbose_gmm, seed=random_seed)
        gmm_params[c] = (means_c, covs_c, weights_c, priors[c])

    log_probs_gmm = np.zeros((X_test_pca.shape[0], n_classes))
    for i, c in enumerate(classes):
        means_c, covs_c, weights_c, prior_c = gmm_params[c]
        log_probs_gmm[:, i] = gmm_log_likelihood(X_test_pca, means_c, covs_c, weights_c) + np.log(prior_c)

    y_pred_gmm = np.argmax(log_probs_gmm, axis=1)
    acc_gmm = accuracy_score(y_test, y_pred_gmm)
    print(f"GMM Model Accuracy: {acc_gmm:.4f}")

    y_test_bin = label_binarize(y_test, classes=classes)

    fpr_gauss = {}
    tpr_gauss = {}
    auc_gauss = {}
    fpr_gmm = {}
    tpr_gmm = {}
    auc_gmm = {}

    for i, c in enumerate(classes):
        fpr_gauss[c], tpr_gauss[c], auc_gauss[c] = compute_roc_auc(y_test_bin[:, i], log_probs_gauss[:, i])
        fpr_gmm[c], tpr_gmm[c], auc_gmm[c] = compute_roc_auc(y_test_bin[:, i], log_probs_gmm[:, i])

    cols = 5
    rows = int(np.ceil(n_classes / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 2.6), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, c in enumerate(classes):
        ax = axes[idx]
        ax.plot(fpr_gauss[c], tpr_gauss[c], linestyle='--', label=f'Gaussian (AUC={auc_gauss[c]:.3f})')
        ax.plot(fpr_gmm[c], tpr_gmm[c], linestyle='-', label=f'GMM (AUC={auc_gmm[c]:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=0.6)
        ax.set_title(f'Class {c}')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', linewidth=0.4)
        ax.legend(loc='lower right', fontsize='x-small')

    for j in range(n_classes, rows * cols):
        fig.delaxes(axes[j])

    fig.suptitle("ROC Curves per class — Gaussian vs GMM", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print("\n===== Comparative Results =====")
    print(f"Gaussian Accuracy: {acc_gauss:.4f}")
    print(f"GMM Accuracy:      {acc_gmm:.4f}")
    print("\nAverage AUC per class:")
    print(f"Gaussian mean AUC: {np.mean(list(auc_gauss.values())):.4f}")
    print(f"GMM mean AUC:      {np.mean(list(auc_gmm.values())):.4f}")

if __name__ == "__main__":
    results = main(random_seed=0,
                   pca_components=40,
                   train_size=10000,
                   test_size=2000,
                   gmm_K=7,
                   gmm_max_iter=50,
                   verbose_gmm=False)