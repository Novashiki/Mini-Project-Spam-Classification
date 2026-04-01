# -*- coding: utf-8 -*-
"""
Spam Detection Using Tree-Based Classifiers
--------------------------------------------
This study evaluates four tree-based machine learning models for binary
classification of SMS messages as spam or ham (legitimate). The models
examined are: Decision Tree, Random Forest, Gradient Boosting, and XGBoost.

Dataset  : SMS Spam Collection v.1 — Almeida, Gomez Hidalgo & Yamakami (2011)
           5,572 messages | 4,825 ham (86.6%) | 747 spam (13.4%)

Feature  : TF-IDF vectorisation (unigrams + bigrams) combined with seven
Extraction  handcrafted numeric features derived from linguistic properties
           of the raw message text.

Evaluation: Accuracy, Precision, Recall, F1-Score, MCC, ROC-AUC

Install  : pip install pandas numpy scipy scikit-learn xgboost matplotlib seaborn
Run      : python spam_detection_complete.py
"""

import os, re, sys, json, string, pickle, warnings
warnings.filterwarnings("ignore")

# Fix for Windows terminals that don't support UTF-8 by default
if hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import scipy.sparse as sp

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — saves plots to disk
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from sklearn.model_selection         import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree                    import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble                import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight      import compute_sample_weight
from sklearn.metrics                 import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
    classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
)
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Output directories — created automatically on first run
# ---------------------------------------------------------------------------
OUT      = "spam_detection_outputs"
OUT_FIGS = os.path.join(OUT, "figures")
OUT_MDL  = os.path.join(OUT, "models")
for d in [OUT, OUT_FIGS, OUT_MDL]:
    os.makedirs(d, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


# ===========================================================================
# SECTION 1 — DATA PIPELINE
# ===========================================================================

# Direct link to the SMS Spam Collection dataset hosted on GitHub.
DATA_URL = (
    "https://raw.githubusercontent.com/justmarkham/"
    "pycon-2016-tutorial/master/data/sms.tsv"
)

# Column names for the 7 handcrafted features added during feature engineering
NUMERIC_FEATURE_COLS = [
    "message_length", "word_count", "digit_count",
    "uppercase_count", "exclamation_count", "currency_count", "url_present",
]


def load_dataset():
    # Columns are named manually: label (ham/spam) and message (raw text).
    df = pd.read_csv(DATA_URL, sep="\t", names=["label", "message"])
    return df


def preprocess_labels(df):
    # Tree-based classifiers require numeric targets. ham =0 ,spam =1
    df = df.copy()
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def clean_text(text):
    # Three-step normalisation applied uniformly before vectorisation.
    # Lowercasing ensures "FREE" and "free" map to the same token.
    # Punctuation removal reduces vocabulary noise.
    # Digit characters are deliberately retained — spam frequently substitutes
    # digits for words (e.g., "2" for "to", "4" for "for").
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def engineer_features(df):
    # Beyond bag-of-words, certain surface-level properties of a message
    # carry strong discriminative signals for spam detection.
    # These seven features were selected based on EDA findings and are
    # consistent with feature sets used in prior SMS spam classification work
    # (Almeida et al., 2011; Gomez Hidalgo et al., 2006).

    df = df.copy()
    msg = df["message"]

    # Character length: spam messages averaged 138.6 chars vs 71.5 for ham (EDA)
    df["message_length"]    = msg.str.len()

    # Word count correlates with length and tends to be higher in spam
    df["word_count"]        = msg.str.split().str.len()

    # Digit frequency: spam uses numeric shortcuts ("txt 2 win", "call 4 free")
    df["digit_count"]       = msg.str.count(r"\d")

    # Uppercase usage: spam employs capital letters for urgency ("FREE", "WIN NOW")
    df["uppercase_count"]   = msg.str.count(r"[A-Z]")

    # Exclamation marks signal urgency — a common rhetorical device in spam
    df["exclamation_count"] = msg.str.count(r"!")

    # Currency symbols indicate prize or money-related content, typical in spam
    df["currency_count"]    = msg.str.count(r"[£$€]")

    # URL presence: spam often directs users to external links
    df["url_present"]       = msg.str.contains(
        r"http|www|\.com", case=False, regex=True).astype(int)

    return df


def build_tfidf(X_text_train, X_text_test, max_features=5000):
    # TF-IDF (Term Frequency-Inverse Document Frequency) converts raw text
    # into a numerical representation where each token's weight reflects
    # its importance within a document relative to the entire corpus.
    
    # Configuration choices:
    #   max_features=5000  — caps vocabulary size; empirically sufficient for
    #                        SMS-scale corpora without sacrificing coverage
    #   ngram_range=(1,2)  — includes bigrams alongside unigrams; phrases such
    #                        as "free entry" and "call now" are stronger spam
    #                        indicators than either constituent word alone
    #   sublinear_tf=True  — applies log(1 + tf) to dampen the effect of
    #                        high-frequency tokens; standard in text classification
    #   min_df=2           — discards tokens appearing in fewer than 2 messages,
    #                        reducing noise from typos and hapax legomena

    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )

    # Critical: vocabulary learned exclusively from training data.
    # Applying fit_transform to test data would constitute data leakage.
    tfidf_train = vec.fit_transform(X_text_train)
    tfidf_test  = vec.transform(X_text_test)

    return tfidf_train, tfidf_test, vec


def combine_features(tfidf_train, tfidf_test, num_train, num_test):
    # The sparse TF-IDF matrix (5000 columns) is horizontally concatenated
    # with the dense numeric feature matrix (7 columns), yielding a unified
    # 5007-dimensional feature space per sample.
    # scipy.sparse.hstack preserves memory efficiency across the pipeline.
    X_train = sp.hstack([tfidf_train, sp.csr_matrix(num_train.values)])
    X_test  = sp.hstack([tfidf_test,  sp.csr_matrix(num_test.values)])
    return X_train, X_test


def get_full_pipeline(max_features=5000):
    # Orchestrates the complete data preparation workflow.
    # The same pipeline output is shared across all four models to ensure
    # a fair and reproducible comparison under identical experimental conditions.

    print("  Loading dataset...")
    df = load_dataset()
    df = preprocess_labels(df)
    df = engineer_features(df)
    df["cleaned_message"] = df["message"].apply(clean_text)

    # Stratified split preserves the original class distribution (86.6% / 13.4%)
    # in both train and test partitions. random_state=42 ensures reproducibility.
    idx_train, idx_test = train_test_split(
        df.index, test_size=0.2, stratify=df["label"], random_state=42,
    )
    tr, te  = df.loc[idx_train], df.loc[idx_test]
    y_train = tr["label"].reset_index(drop=True)
    y_test  = te["label"].reset_index(drop=True)

    tfidf_tr, tfidf_te, vectorizer = build_tfidf(
        tr["cleaned_message"], te["cleaned_message"], max_features)
    X_train, X_test = combine_features(
        tfidf_tr, tfidf_te,
        tr[NUMERIC_FEATURE_COLS], te[NUMERIC_FEATURE_COLS])

    return X_train, X_test, y_train, y_test, vectorizer, df


# ===========================================================================
# SECTION 2 — EVALUATION METRICS
# ===========================================================================

def evaluate_model(model_name, y_true, y_pred, y_proba=None):
    # Six metrics are computed to provide a comprehensive view of performance.
    # Accuracy alone is insufficient given the class imbalance in this dataset.
    # A classifier that predicts ham for every message would achieve 86.6%
    # accuracy while detecting zero spam — a meaningless result.
    #
    # F1-Score balances precision and recall and is the primary metric here.
    # MCC (Matthews Correlation Coefficient) is included as it accounts for
    # all four quadrants of the confusion matrix and is robust to imbalance.
    # (Chicco & Jurman, 2020, BMC Genomics)

    return {
        "model":     model_name,
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score":  round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "mcc":       round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "roc_auc":   round(float(roc_auc_score(y_true, y_proba)), 4) if y_proba is not None else None,
    }


def print_report(model_name, y_true, y_pred, y_proba=None):
    m  = evaluate_model(model_name, y_true, y_pred, y_proba)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*55}")
    print(f"  Model     : {model_name}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1-Score  : {m['f1_score']:.4f}")
    print(f"  MCC       : {m['mcc']:.4f}")
    if m["roc_auc"]:
        print(f"  ROC-AUC   : {m['roc_auc']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted Ham   Predicted Spam")
    print(f"  Actual Ham      {cm[0,0]:^14}  {cm[0,1]:^14}")
    print(f"  Actual Spam     {cm[1,0]:^14}  {cm[1,1]:^14}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["ham", "spam"], digits=4))
    return m


def metrics_dataframe(metrics_list):
    return pd.DataFrame(metrics_list).set_index("model")


# ===========================================================================
# SECTION 3 — TREE-BASED CLASSIFIERS
# ===========================================================================


def build_decision_tree(X_train, X_test, y_train, y_test, vectorizer):
    # Decision Tree (CART — Breiman et al., 1984)
    # A non-parametric supervised learner that recursively partitions the
    # feature space by selecting the split that maximises the reduction in
    # Gini impurity at each internal node.
    #
    # It serves as the baseline interpretable model in this comparison and
    # is the foundational component of the ensemble methods evaluated below.
    # max_depth=20 prevents unconstrained growth while allowing sufficient
    # expressiveness for the high-dimensional TF-IDF feature space.

    print("\n" + "="*60)
    print("  MODEL 1 : Decision Tree  (Breiman et al., 1984)")
    print("="*60)

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )
    print("  Training...")
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = print_report("Decision Tree", y_test, y_pred, y_proba)

    path = os.path.join(OUT_MDL, "decision_tree.pkl")
    with open(path, "wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer}, f)
    print(f"  Saved -> {path}")
    return model, metrics, y_pred, y_proba


def build_random_forest(X_train, X_test, y_train, y_test, vectorizer):
    # Random Forest (Breiman, 2001)
    # An ensemble of B decision trees, each trained on an independently
    # drawn bootstrap sample. At each split, only a random subset of
    # sqrt(p) features is considered, decorrelating the trees and reducing
    # ensemble variance beyond what bagging alone achieves.
    #
    # The aggregated prediction (majority vote / averaged probabilities)
    # is substantially more stable than any individual tree, which is
    # why Random Forest typically outperforms Decision Tree on unseen data.
    # n_estimators=200 provides a good bias-variance tradeoff for this corpus.

    print("\n" + "="*60)
    print("  MODEL 2 : Random Forest  (Breiman, 2001)")
    print("="*60)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    print("  Training...")
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = print_report("Random Forest", y_test, y_pred, y_proba)

    path = os.path.join(OUT_MDL, "random_forest.pkl")
    with open(path, "wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer}, f)
    print(f"  Saved -> {path}")
    return model, metrics, y_pred, y_proba


def build_gradient_boosting(X_train, X_test, y_train, y_test, vectorizer):
    # Gradient Boosting Machine (Friedman, 2001)
    # Unlike Random Forest, which builds trees in parallel, GBM constructs
    # an additive model sequentially. Each new tree is fit to the negative
    # gradient (pseudo-residuals) of the log-loss function, correcting the
    # errors of the current ensemble.
    #
    # learning_rate=0.1 (shrinkage) scales each tree's contribution,
    # reducing overfitting when combined with a sufficient number of trees.
    # subsample=0.8 introduces stochasticity (Stochastic GBM — Friedman, 2002),
    # further improving generalisation on the held-out test set.
    # max_depth=4 keeps individual learners weak, as required by boosting theory.

    print("\n" + "="*60)
    print("  MODEL 3 : Gradient Boosting  (Friedman, 2001)")
    print("="*60)

    # GradientBoostingClassifier does not accept class_weight.
    # Equivalent weighting is achieved by passing per-sample weights at fit().
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=2,
        random_state=42,
    )
    print("  Training...")
    model.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = print_report("Gradient Boosting", y_test, y_pred, y_proba)

    path = os.path.join(OUT_MDL, "gradient_boosting.pkl")
    with open(path, "wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer}, f)
    print(f"  Saved -> {path}")
    return model, metrics, y_pred, y_proba


def build_xgboost(X_train, X_test, y_train, y_test, vectorizer):
    # XGBoost — eXtreme Gradient Boosting (Chen & Guestrin, 2016)
    # scale_pos_weight = n_ham / n_spam re-weights the gradient updates

    print("\n" + "="*60)
    print("  MODEL 4 : XGBoost  (Chen & Guestrin, 2016)")
    print("="*60)

    neg = int((y_train == 0).sum())   # n_ham
    pos = int((y_train == 1).sum())   # n_spam
    spw = round(neg / pos, 2)
    print(f"  n_ham={neg}  n_spam={pos}  scale_pos_weight={spw}")

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    print("  Training...")
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = print_report("XGBoost", y_test, y_pred, y_proba)

    path = os.path.join(OUT_MDL, "xgboost.pkl")
    with open(path, "wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer}, f)
    print(f"  Saved -> {path}")
    return model, metrics, y_pred, y_proba

# ===========================================================================
# SECTION 4 — VISUALISATIONS
# ===========================================================================

def plot_confusion_matrix(model_name, y_true, y_pred):
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels = np.array([
        [f"{cm[i,j]}\n({cm_pct[i,j]:.1%})" for j in range(2)]
        for i in range(2)
    ])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_pct, annot=labels, fmt="", cmap="Blues",
                xticklabels=["Ham", "Spam"],
                yticklabels=["Ham", "Spam"],
                linewidths=0.5, ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label",      fontweight="bold")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(OUT_FIGS, f"confusion_matrix_{model_name.replace(' ','_').lower()}.png")
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  [OK] {fname}")


def plot_roc_curves(results):
    # ROC curves plot TPR against FPR across all decision thresholds.
    # AUC summarises the curve into a single scalar; a value of 1.0
    # indicates perfect separability. The diagonal represents a random
    # classifier (AUC = 0.5) and serves as the lower-bound reference.
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, r in enumerate(results):
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_proba"])
        ax.plot(fpr, tpr, color=COLORS[i], lw=2,
                label=f"{r['name']}  (AUC = {auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier (AUC = 0.5)")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate",  fontweight="bold")
    ax.set_title("ROC Curves — Tree-Based Classifiers", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fname = os.path.join(OUT_FIGS, "roc_curves_all_models.png")
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  [OK] {fname}")


def plot_precision_recall(results):
    # Precision-Recall curves are a more informative diagnostic than ROC
    # for imbalanced classification tasks (Saito & Rehmsmeier, 2015, PLOS ONE).
    # The dashed horizontal line marks the no-skill baseline, equal to the
    # proportion of positive (spam) samples in the test set (~13.4%).
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, r in enumerate(results):
        prec, rec, _ = precision_recall_curve(r["y_true"], r["y_proba"])
        ap = average_precision_score(r["y_true"], r["y_proba"])
        ax.plot(rec, prec, color=COLORS[i], lw=2,
                label=f"{r['name']}  (AP = {ap:.4f})")
    baseline = float(np.mean(results[0]["y_true"]))
    ax.axhline(y=baseline, color="gray", linestyle="--", lw=1,
               label=f"No-skill baseline ({baseline:.2%})")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("Recall",    fontweight="bold")
    ax.set_ylabel("Precision", fontweight="bold")
    ax.set_title("Precision-Recall Curves — Tree-Based Classifiers",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fname = os.path.join(OUT_FIGS, "precision_recall_curves.png")
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  [OK] {fname}")


def plot_metrics_comparison(metrics_list):
    # Grouped bar chart for direct cross-model comparison across all six metrics.
    keys   = ["accuracy", "precision", "recall", "f1_score", "mcc", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score", "MCC", "ROC-AUC"]
    names  = [m["model"] for m in metrics_list]
    n      = len(names)
    x      = np.arange(len(keys))
    w      = 0.8 / n
    off    = -(n - 1) / 2

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (m, name) in enumerate(zip(metrics_list, names)):
        vals = [m.get(k) or 0 for k in keys]
        bars = ax.bar(x + (off + i) * w, vals, w, label=name,
                      color=COLORS[i], alpha=0.88, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7.5, rotation=45)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Performance Comparison — Tree-Based Classifiers",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.axhline(y=0.95, color="grey", linestyle="--", lw=0.8, alpha=0.5)
    plt.tight_layout()
    fname = os.path.join(OUT_FIGS, "metrics_comparison.png")
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  [OK] {fname}")


def plot_feature_importance(model, model_name, vectorizer, top_n=25):
    # Feature importance is measured by mean decrease in Gini impurity
    # across all trees in the ensemble. This gives an indication of which
    # tokens and numeric features the model relied on most for its decisions.
    # TF-IDF feature names are retrieved from the fitted vectorizer vocabulary.
    all_names = (
        vectorizer.get_feature_names_out().tolist() +
        ["message_length", "word_count", "digit_count",
         "uppercase_count", "exclamation_count", "currency_count", "url_present"]
    )
    importances = model.feature_importances_
    min_len     = min(len(all_names), len(importances))
    all_names   = all_names[:min_len]
    importances = importances[:min_len]

    idx        = np.argsort(importances)[::-1][:top_n]
    top_names  = [all_names[i]   for i in idx]
    top_values = [importances[i] for i in idx]

    fig, ax = plt.subplots(figsize=(9, max(6, top_n * 0.32)))
    colors  = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))[::-1]
    ax.barh(range(top_n), top_values[::-1], color=colors[::-1], edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Mean Decrease in Gini Impurity", fontweight="bold")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}",
                 fontsize=13, fontweight="bold")
    for j, val in enumerate(top_values[::-1]):
        ax.text(val + 0.0002, j, f"{val:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    fname = os.path.join(OUT_FIGS, f"feature_importance_{model_name.replace(' ','_').lower()}.png")
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  [OK] {fname}")


def plot_decision_tree(dt_model, vectorizer, max_depth=4):
    # The full Decision Tree is visualised to a depth of 4 for readability.
    # Each node displays: the splitting feature and threshold, Gini impurity,
    # sample count, and class distribution. Leaf colour intensity reflects
    # class confidence (blue = ham, orange = spam).
    feature_names = (
        vectorizer.get_feature_names_out().tolist() +
        ["message_length", "word_count", "digit_count",
         "uppercase_count", "exclamation_count", "currency_count", "url_present"]
    )
    print("\n  Decision Tree — text representation (depth <= 4):\n")
    print(export_text(dt_model, feature_names=feature_names, max_depth=max_depth))

    fig, ax = plt.subplots(figsize=(22, 8))
    plot_tree(dt_model, max_depth=max_depth, feature_names=feature_names,
              class_names=["ham", "spam"], filled=True, rounded=True,
              fontsize=8, ax=ax, impurity=True, proportion=False)
    ax.set_title(f"Decision Tree Structure (depth shown: {max_depth})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(OUT_FIGS, "decision_tree_diagram.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [OK] {fname}")


def plot_message_length_dist(df):
    # Distribution of message character length for each class.

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (lbl, name) in zip(axes, {0: "Ham", 1: "Spam"}.items()):
        subset = df[df["label"] == lbl]["message_length"]
        color  = "#4C72B0" if name == "Ham" else "#C44E52"
        sns.histplot(subset, kde=True, ax=ax, color=color, alpha=0.6, bins=40)
        ax.axvline(subset.mean(), color="black", linestyle="--", lw=1.5,
                   label=f"Mean = {subset.mean():.1f} chars")
        ax.set_xlabel("Message Length (characters)", fontweight="bold")
        ax.set_ylabel("Frequency", fontweight="bold")
        ax.set_title(f"Length Distribution — {name}", fontsize=12, fontweight="bold")
        ax.legend()
    plt.suptitle("SMS Message Length: Ham vs. Spam",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fname = os.path.join(OUT_FIGS, "message_length_distribution.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [OK] {fname}")

# ===========================================================================
# SECTION 5 — INFERENCE
# ===========================================================================

def predict_message(message, model_name="xgboost"):
    # Loads a serialised model and applies the identical preprocessing steps
    # used during training. Consistency between training and inference
    # transformations is essential; any deviation would produce a feature
    # vector incompatible with the model's learned decision boundaries.

    path = os.path.join(OUT_MDL, f"{model_name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model at {path}. Run main() first.")

    with open(path, "rb") as f:
        bundle = pickle.load(f)
    model, vec = bundle["model"], bundle["vectorizer"]

    cleaned   = clean_text(message)
    tfidf_vec = vec.transform([cleaned])
    num_feats = np.array([[
        len(message),
        len(message.split()),
        sum(c.isdigit() for c in message),
        sum(c.isupper() for c in message),
        message.count("!"),
        sum(message.count(c) for c in "£$€"),
        int(bool(re.search(r"http|www|\.com", message, re.I))),
    ]])
    X     = sp.hstack([tfidf_vec, sp.csr_matrix(num_feats)])
    pred  = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    return {
        "model":            model_name,
        "label":            "SPAM" if pred == 1 else "HAM",
        "spam_probability": round(float(proba[1]), 4),
        "ham_probability":  round(float(proba[0]), 4),
        "confidence":       round(float(max(proba)), 4),
    }


def predict_all_models(message):
    # Runs inference across all four trained models for direct comparison.
    print(f'\n  Input: "{message}"\n')
    print(f"  {'Model':<22}  {'Label':<6}  {'P(spam)':>9}  {'Confidence':>11}")
    print("  " + "-"*54)
    for name in ["decision_tree", "random_forest", "gradient_boosting", "xgboost"]:
        try:
            r = predict_message(message, name)
            print(f"  {r['model']:<22}  {r['label']:<6}  "
                  f"{r['spam_probability']:>9.4f}  {r['confidence']:>11.4f}")
        except FileNotFoundError:
            print(f"  {name:<22}  [model not found — run main() first]")
    print()

# ===========================================================================
# SECTION 6 — MAIN
# ===========================================================================

def main():
    print("\n" + "#"*60)
    print("  SPAM DETECTION — TREE-BASED CLASSIFIERS")
    print(f"  Output directory: {os.path.abspath(OUT)}/")
    print("#"*60)

    # Data loaded once and shared across all models for a fair comparison
    print("\n[STEP 1] Data preparation...")
    X_train, X_test, y_train, y_test, vectorizer, df = get_full_pipeline()
    print(f"  Total : {len(df):,} messages  |  Train : {X_train.shape[0]:,}  |  Test : {X_test.shape[0]:,}")
    print(f"  Features : {X_train.shape[1]}  |  Spam prevalence (train) : {y_train.mean():.2%}")

    print("\n[STEP 2] EDA — message length distribution...")
    plot_message_length_dist(df)

    print("\n[STEP 3] Training classifiers...")
    dt_model,  dt_m,  dt_yp,  dt_ypr  = build_decision_tree(X_train, X_test, y_train, y_test, vectorizer)
    rf_model,  rf_m,  rf_yp,  rf_ypr  = build_random_forest(X_train, X_test, y_train, y_test, vectorizer)
    gb_model,  gb_m,  gb_yp,  gb_ypr  = build_gradient_boosting(X_train, X_test, y_train, y_test, vectorizer)
    xgb_model, xgb_m, xgb_yp, xgb_ypr = build_xgboost(X_train, X_test, y_train, y_test, vectorizer)

    all_metrics = [dt_m, rf_m, gb_m, xgb_m]

    print("\n[STEP 4] Saving results...")
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    df_metrics = metrics_dataframe(all_metrics)
    df_metrics.to_csv(os.path.join(OUT, "comparison.csv"))

    print("\n" + "="*60)
    print("  COMPARATIVE RESULTS")
    print("="*60)
    print(df_metrics.to_string())
    print(f"\n  Best F1-Score : {df_metrics['f1_score'].idxmax()}  ({df_metrics['f1_score'].max():.4f})")
    print(f"  Best ROC-AUC  : {df_metrics['roc_auc'].idxmax()}  ({df_metrics['roc_auc'].max():.4f})")
    print(f"  Best Accuracy : {df_metrics['accuracy'].idxmax()}  ({df_metrics['accuracy'].max():.4f})")

    print("\n[STEP 5] Generating figures...")
    roc_input = [
        {"name": "Decision Tree",     "y_true": y_test, "y_proba": dt_ypr},
        {"name": "Random Forest",     "y_true": y_test, "y_proba": rf_ypr},
        {"name": "Gradient Boosting", "y_true": y_test, "y_proba": gb_ypr},
        {"name": "XGBoost",           "y_true": y_test, "y_proba": xgb_ypr},
    ]
    plot_confusion_matrix("Decision Tree",     y_test, dt_yp)
    plot_confusion_matrix("Random Forest",     y_test, rf_yp)
    plot_confusion_matrix("Gradient Boosting", y_test, gb_yp)
    plot_confusion_matrix("XGBoost",           y_test, xgb_yp)
    plot_roc_curves(roc_input)
    plot_precision_recall(roc_input)
    plot_metrics_comparison(all_metrics)
    plot_feature_importance(rf_model,  "Random Forest",     vectorizer, top_n=25)
    plot_feature_importance(gb_model,  "Gradient Boosting", vectorizer, top_n=25)
    plot_feature_importance(xgb_model, "XGBoost",           vectorizer, top_n=25)
    plot_decision_tree(dt_model, vectorizer, max_depth=4)

    print("\n[STEP 6] Inference on sample messages...")
    predict_all_models("FREE entry! Win £1000 cash prize. Call NOW to claim!!!")
    predict_all_models("Hey, are you coming to the study group tonight?")

    print("\n" + "#"*60)
    print("  COMPLETE")
    print("#"*60)
    print(f"  Figures  -> {os.path.abspath(OUT_FIGS)}/")
    print(f"  Models   -> {os.path.abspath(OUT_MDL)}/")
    print(f"  JSON     -> {os.path.abspath(OUT)}/results.json")
    print(f"  CSV      -> {os.path.abspath(OUT)}/comparison.csv")

if __name__ == "__main__":
    main()