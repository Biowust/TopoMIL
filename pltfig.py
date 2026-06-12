

import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)


class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)

def load_oof_scores(pkl_path):
    with open(pkl_path, "rb") as f:
        data = NumpyCompatUnpickler(f).load()

    if "y_true_folds" not in data or "y_score_folds" not in data:
        raise ValueError(f"{pkl_path} does not contain y_true_folds / y_score_folds")

    y_true_folds = [np.asarray(x).astype(int) for x in data["y_true_folds"]]
    y_score_folds = [np.asarray(x).astype(float) for x in data["y_score_folds"]]

    if len(y_true_folds) != len(y_score_folds):
        raise ValueError(f"{pkl_path}: len(y_true_folds) != len(y_score_folds)")

    return y_true_folds, y_score_folds


def merge_folds(y_true_folds, y_score_folds):
    y_true = np.concatenate(y_true_folds, axis=0)
    y_score = np.concatenate(y_score_folds, axis=0)
    return y_true, y_score


def interp_unique(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x_unique, idx = np.unique(x, return_index=True)
    y_unique = y[idx]
    return x_unique, y_unique


def compute_mean_roc(y_true_folds, y_score_folds, n_grid=1001):
    mean_fpr = np.linspace(0.0, 1.0, n_grid)
    tprs = []
    aucs = []

    for yt, ys in zip(y_true_folds, y_score_folds):
        fpr, tpr, _ = roc_curve(yt, ys, pos_label=1)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        tpr_i = np.interp(mean_fpr, fpr, tpr)
        tpr_i[0] = 0.0
        tprs.append(tpr_i)

    tprs = np.vstack(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    return mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc


def compute_mean_pr(y_true_folds, y_score_folds, n_grid=1001):
    recall_grid = np.linspace(0.0, 1.0, n_grid)
    precisions_interp = []
    aps = []

    for yt, ys in zip(y_true_folds, y_score_folds):
        precision, recall, _ = precision_recall_curve(yt, ys, pos_label=1)
        ap = average_precision_score(yt, ys)
        aps.append(ap)

        recall_u, precision_u = interp_unique(recall, precision)
        p_i = np.interp(recall_grid, recall_u, precision_u)
        precisions_interp.append(p_i)

    precisions_interp = np.vstack(precisions_interp)
    mean_p = precisions_interp.mean(axis=0)
    std_p = precisions_interp.std(axis=0)
    mean_ap = np.mean(aps)
    std_ap = np.std(aps)

    return recall_grid, mean_p, std_p, mean_ap, std_ap


def plot_multi_model_roc_pooled(model_results, save_path):
    plt.figure(figsize=(6, 5))

    for model_name, pack in model_results.items():
        y_true, y_score = merge_folds(pack["y_true_folds"], pack["y_score_folds"])
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison (Pooled OOF)")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_multi_model_pr_pooled(model_results, save_path, show_baseline=True):
    plt.figure(figsize=(6, 5))

    baseline_drawn = False
    for model_name, pack in model_results.items():
        y_true, y_score = merge_folds(pack["y_true_folds"], pack["y_score_folds"])
        precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=1)
        ap = average_precision_score(y_true, y_score)
        plt.plot(recall, precision, lw=2, label=f"{model_name} (AP={ap:.3f})")

        if show_baseline and not baseline_drawn:
            pos_rate = float(np.mean(y_true))
            plt.axhline(y=pos_rate, linestyle="--", color="gray", lw=1,
                        label=f"Random baseline ({pos_rate:.3f})")
            baseline_drawn = True

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve Comparison (Pooled OOF)")
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_multi_model_roc_meanstd(model_results, save_path, shade_models=None):
    if shade_models is None:
        shade_models = []

    plt.figure(figsize=(6, 5))

    for model_name, pack in model_results.items():
        mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc = compute_mean_roc(
            pack["y_true_folds"], pack["y_score_folds"]
        )

        plt.plot(mean_fpr, mean_tpr, lw=2,
                 label=f"{model_name} (AUC={mean_auc:.3f}±{std_auc:.3f})")

        if model_name in shade_models:
            lo = np.clip(mean_tpr - std_tpr, 0.0, 1.0)
            hi = np.clip(mean_tpr + std_tpr, 0.0, 1.0)
            plt.fill_between(mean_fpr, lo, hi, alpha=0.15)

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison (5-fold OOF Mean±Std)")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_multi_model_pr_meanstd(model_results, save_path, shade_models=None, show_baseline=True):
    if shade_models is None:
        shade_models = []

    plt.figure(figsize=(6, 5))

    baseline_drawn = False
    for model_name, pack in model_results.items():
        recall_grid, mean_p, std_p, mean_ap, std_ap = compute_mean_pr(
            pack["y_true_folds"], pack["y_score_folds"]
        )

        plt.plot(recall_grid, mean_p, lw=2,
                 label=f"{model_name} (AP={mean_ap:.3f}±{std_ap:.3f})")

        if model_name in shade_models:
            lo = np.clip(mean_p - std_p, 0.0, 1.0)
            hi = np.clip(mean_p + std_p, 0.0, 1.0)
            plt.fill_between(recall_grid, lo, hi, alpha=0.15)

        if show_baseline and not baseline_drawn:
            y_true, _ = merge_folds(pack["y_true_folds"], pack["y_score_folds"])
            pos_rate = float(np.mean(y_true))
            plt.axhline(y=pos_rate, linestyle="--", color="gray", lw=1,
                        label=f"Random baseline ({pos_rate:.3f})")
            baseline_drawn = True

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve Comparison (5-fold OOF Mean±Std)")
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def parse_input_pairs(input_list):
    """
    """
    results = {}
    for item in input_list:
        if "=" not in item:
            raise ValueError(f"Invalid input format: {item}, expected NAME=PATH")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        results[name] = path
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=False,
        default=[
            "FRMIL=/mnt/data/ljc/outputs/frmil_baseline/crc/2026_March_16_16h_13m_mamba/crc_oof_scores.pkl",
            "DSMIL=/mnt/data/ljc/outputs/dsmil_crc/crc_oof_scores.pkl",
            "TransMIL=/mnt/data/ljc/outputs/transmil_crc/crc_oof_scores.pkl",
            "TopoMIL(Ours)=/mnt/data/ljc/outputs/frmil/crc/crc_oof_scores.pkl"
        ],
        help="Format: FRMIL=/path/to/crc_oof_scores.pkl DSMIL=/path/to/... Ours=/path/to/..."
    )
    parser.add_argument("--save_dir", type=str, default="/mnt/data/ljc/outputs")
    parser.add_argument("--shade_models", nargs="*", default="TopoMIL(Ours)",
                        help="Models to show std shading, e.g. --shade_models Ours")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    input_map = parse_input_pairs(args.inputs)

    model_results = {}
    for model_name, pkl_path in input_map.items():
        y_true_folds, y_score_folds = load_oof_scores(pkl_path)
        model_results[model_name] = {
            "y_true_folds": y_true_folds,
            "y_score_folds": y_score_folds,
        }
        print(f"[Loaded] {model_name}: {pkl_path}")

    # pooled
    roc_pooled_path = os.path.join(args.save_dir, "crc_multi_roc_pooled.png")
    pr_pooled_path = os.path.join(args.save_dir, "crc_multi_pr_pooled.png")
    plot_multi_model_roc_pooled(model_results, roc_pooled_path)
    plot_multi_model_pr_pooled(model_results, pr_pooled_path)

    # mean ± std
    roc_meanstd_path = os.path.join(args.save_dir, "crc_multi_roc_meanstd.png")
    pr_meanstd_path = os.path.join(args.save_dir, "crc_multi_pr_meanstd.png")
    plot_multi_model_roc_meanstd(model_results, roc_meanstd_path, shade_models=args.shade_models)
    plot_multi_model_pr_meanstd(model_results, pr_meanstd_path, shade_models=args.shade_models)

    print(f"[Saved] {roc_pooled_path}")
    print(f"[Saved] {pr_pooled_path}")
    print(f"[Saved] {roc_meanstd_path}")
    print(f"[Saved] {pr_meanstd_path}")


if __name__ == "__main__":
    main() 