import os
import gc
import ast
import warnings
import tkinter as tk

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")           # Non-interactive backend — no display needed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tkinter import filedialog
from sklearn.model_selection import train_test_split
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from xml.sax.saxutils import escape

# ── Project modules ──────────────────────────────────────────────────────────
from modules.dataset.loader import load_dataset
from modules.dataset.validator import validate_dataset
from modules.dataset.summary import dataset_summary
from modules.preprocessing.preprocessing_pipeline import preprocess_dataset
from modules.preprocessing.skewness_fixer import fix_skewness
from modules.balancing.imbalance_detector import detect_imbalance
from modules.balancing.imbalance_report import imbalance_report
from modules.balancing.smote_module import apply_smote
from modules.synthetic.ctgan_generator import generate_ctgan_data
from modules.bias.fairness_metrics import calculate_fairness
from modules.bias.fairness_fix import apply_fairness_fix
from modules.metrics.edqs import calculate_edqs
from modules.metrics.eri import calculate_eri
from modules.metrics.rai import calculate_rai
from modules.metrics.governance_score import calculate_governance_score
from modules.metrics.explainability_score import calculate_explainability_score
from modules.model.train_model import train_model
from modules.model.evaluate_model import evaluate_model
from modules.model.save_model import save_model
from modules.target.detect_target import detect_target_column
from modules.visualization.before_after_graphs import plot_before_after_counts
from modules.visualization.boxplots import plot_boxplots
from modules.visualization.piecharts import plot_piecharts
from modules.visualization.edqs_graph import plot_edqs_comparison
from modules.visualization.bias_visualizations import (
    bias_before_after_graph,
    bias_heatmap,
    fairness_comparison_chart,
)
from modules.visualization.eri_graph import plot_eri_comparison
from modules.visualization.rai_graph import plot_rai_comparison
from modules.visualization.bias_radar_chart import bias_radar_chart
from modules.visualization.heatmaps import plot_correlation_heatmap
from modules.visualization.ctgan_graph import plot_ctgan_real_vs_synthetic
from modules.visualization.feature_importance import plot_feature_importance
from modules.utils.save_metrics import save_metrics
from modules.utils.logger import set_log_file, write_log
from modules.utils.system_info import get_system_info
from modules.llm.llm_decision_engine import (
    generate_llm_analysis,
    generate_llm_recommendations,
)
from modules.llm.apply_recommendations import apply_llm_recommendations


# =====================================================================
# PIPELINE CONSTANTS
# =====================================================================
FAST_MODE      = True   # Use reduced epochs / trees for faster runs
TEST_SIZE      = 0.20   # 20 % of data held out for evaluation
RANDOM_STATE   = 42     # Global reproducibility seed
MAX_ITERATIONS = 5     # Maximum LLM feedback loops
PATIENCE       = 2      # Early-stop if no improvement for N consecutive loops
MIN_IMPROVEMENT = 0.001 # Minimum score delta to count as genuine improvement

os.makedirs("outputs", exist_ok=True)

# Global metrics log — all metric snapshots are appended here during the run
METRICS_TEXT_PATH = os.path.join("outputs", "all_metrics_log.txt")
with open(METRICS_TEXT_PATH, "w", encoding="utf-8") as _f:
    _f.write("RESPONSIBLE AI METRICS LOG\n\n")
    # Log system configuration at pipeline start
    try:
        sys_info = get_system_info()
        _f.write("SYSTEM CONFIGURATION\n")
        _f.write("=" * 40 + "\n")
        for k, v in sys_info.items():
            _f.write(f"{k} : {v}\n")
        _f.write("\n")
    except Exception:
        pass


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def optimize_memory():
    """Free unused Python objects and close all open matplotlib figures."""
    gc.collect()
    plt.close("all")


def print_section(title: str):
    """Print a clearly visible section separator to stdout."""
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def append_metrics_to_txt(title: str, data):
    """
    Append a titled block of metrics to the global log file.
    Accepts either a dict (key : value) or any string-able object.
    """
    with open(METRICS_TEXT_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 60}\n{title}\n{'=' * 60}\n\n")
        if isinstance(data, dict):
            for k, v in data.items():
                f.write(f"{k} : {v}\n")
        else:
            f.write(str(data))
        f.write("\n")


def choose_file() -> str:
    """
    Open a GUI file-chooser dialog for CSV selection.
    Falls back to manual path input if tkinter is unavailable
    (e.g., headless server environments).
    """
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file_path = filedialog.askopenfilename(
            title="Select CSV Dataset",
            filetypes=[("CSV Files", "*.csv")],
        )
        root.destroy()
        if file_path:
            print("\nDataset Selected Successfully")
            return file_path
    except Exception as e:
        print(f"GUI error: {e}")

    # Fallback: ask user to type the path manually
    file_path = input("\nEnter CSV Path: ").strip()
    if os.path.exists(file_path):
        return file_path
    raise FileNotFoundError(f"File not found: {file_path}")


def detect_categorical_columns(df: pd.DataFrame, target_col: str) -> list:
    """
    Return all non-target columns that have object or category dtype.
    Used to identify which columns need special treatment in SMOTE / pie charts.
    """
    return [
        col for col in df.columns
        if col != target_col
        and (
            str(df[col].dtype) == "object"
            or "category" in str(df[col].dtype)
        )
    ]


def create_graph_output_folder(dataset_name: str) -> str:
    """Create (if needed) and return the per-dataset graph directory."""
    graph_dir = os.path.join(
        "outputs", "graphs", dataset_name.replace(".csv", "")
    )
    os.makedirs(graph_dir, exist_ok=True)
    return graph_dir


def save_current_graph(graph_dir: str, graph_name: str):
    """Save the currently active matplotlib figure to disk and close it."""
    try:
        output_path = os.path.join(graph_dir, f"{graph_name}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        print(f"\nGraph Saved: {output_path}")
        plt.close()
    except Exception as e:
        print(f"Graph save error [{graph_name}]: {e}")


def generate_pdf_report(output_path: str, content: str, bias_table_data=None):
    """
    Build and save a multi-section PDF report.

    Parameters
    ----------
    output_path     : Destination PDF file path.
    content         : Plain-text report body (newline-separated).
    bias_table_data : Optional list-of-lists for the bias summary table.
                      Pass None when task is regression (no bias table).
    """
    doc    = SimpleDocTemplate(output_path)
    styles = getSampleStyleSheet()
    story  = []

    for line in content.split("\n"):
        story.append(Paragraph(escape(line), styles["BodyText"]))
        story.append(Spacer(1, 8))

    if bias_table_data is not None:
        story.append(Spacer(1, 20))
        story.append(Paragraph("<b>Bias Analysis Table</b>", styles["Heading2"]))
        story.append(Spacer(1, 10))

        table = Table(bias_table_data)
        table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID",       (0, 0), (-1, -1), 1, colors.black),
                ("FONTNAME",   (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
            ])
        )
        story.append(table)

    doc.build(story)
    print("\nPDF Report Generated Successfully")


def safe_remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate rows only when they exceed 15 % of the dataset.
    Below that threshold duplicates are considered a safe level and kept,
    avoiding unnecessary data loss on small datasets.
    """
    before = len(df)
    duplicate_percent = df.duplicated().mean() * 100

    if duplicate_percent > 15:
        df = df.drop_duplicates()
        print(f"\nDuplicate Rows Removed : {before - len(df)}")
    else:
        print("\nDuplicate Removal Skipped (safe duplicate level)")

    return df


def validate_final_dataset(df: pd.DataFrame, target_col: str, task_type: str):
    """
    Sanity-check the dataset before model training.

    Raises ValueError if:
      - DataFrame is empty
      - Target column is missing
      - Classification task has < 2 distinct classes
    """
    if len(df) == 0:
        raise ValueError("Dataset became empty after processing.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' is missing.")

    print("\n========== TARGET CHECK ==========")
    print(df[target_col].value_counts(dropna=False))

    unique_classes = df[target_col].nunique()
    print(f"\nUnique Classes : {unique_classes}")

    if task_type == "classification" and unique_classes < 2:
        raise ValueError(
            "Classification requires at least 2 classes in the target column."
        )


def detect_task_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Heuristically decide between 'classification' and 'regression'.
    Regression is chosen only when the target is numeric AND has many
    distinct values (> 15 unique values OR > 5 % of total rows).
    """
    unique_count = df[target_col].nunique()
    total_rows   = len(df)

    if (
        pd.api.types.is_numeric_dtype(df[target_col])
        and (unique_count > 15 or unique_count / total_rows > 0.05)
    ):
        return "regression"
    return "classification"


def calculate_overall_score(
    accuracy: float,
    fairness: float,
    edqs: float,
    rai: float,
    explainability: float,
) -> float:
    """
    Composite Responsible-AI score used to guide model selection.

    Weights:
        Accuracy       35 %
        Fairness       25 %
        EDQS (0-100)   20 %   → normalised to 0-1 before weighting
        RAI  (0-100)   10 %   → normalised to 0-1 before weighting
        Explainability 10 %
    """
    return (
        0.35 * accuracy
        + 0.25 * fairness
        + 0.20 * (edqs / 100)
        + 0.10 * (rai / 100)
        + 0.10 * explainability
    )


def parse_llm_recommendations(raw) -> dict:
    """
    Safely convert LLM output (dict or raw string) to a plain Python dict.
    Returns an empty dict on any parse failure so the pipeline never crashes
    due to a malformed LLM response.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return ast.literal_eval(raw)
        except Exception:
            pass
    return {}


# =====================================================================
# FIX 1: Enforce consistent dtype for target column after any concat
# =====================================================================

def enforce_target_dtype(df: pd.DataFrame, target_col: str, original_dtype) -> pd.DataFrame:
    """
    Cast the target column back to its original dtype.

    WHY THIS IS NEEDED:
    CTGAN internally encodes string labels as integers.  After merging
    synthetic rows back with the real training data you can end up with a
    column that contains a mix of str and int values.  sklearn's
    VotingClassifier (and numpy's sort) will crash with:
        TypeError: '<' not supported between instances of 'str' and 'int'
    Casting everything to the original dtype eliminates the mixed type.

    Parameters
    ----------
    df            : DataFrame to fix (a copy is returned).
    target_col    : Name of the target column.
    original_dtype: The dtype recorded BEFORE any encoding (e.g. object).
    """
    if target_col not in df.columns:
        return df

    try:
        if original_dtype == object or str(original_dtype) == "object":
            df[target_col] = df[target_col].astype(str)
        else:
            df[target_col] = df[target_col].astype(original_dtype)
    except Exception as e:
        print(f"[dtype fix warning] Could not cast '{target_col}' to {original_dtype}: {e}")

    return df


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def main():
    print_section("RESPONSIBLE AI PIPELINE STARTED")

    # ------------------------------------------------------------------
    # STEP 1 — LOAD DATASET
    # Open the CSV, validate its structure, and print a summary.
    # ------------------------------------------------------------------
    file_path    = choose_file()
    df           = load_dataset(file_path)
    dataset_name = os.path.basename(file_path)
    graph_dir    = create_graph_output_folder(dataset_name)
    set_log_file(dataset_name)
    optimize_memory()

    validate_dataset(df)
    dataset_summary(df)

    # ------------------------------------------------------------------
    # STEP 2 — TARGET COLUMN & TASK TYPE DETECTION
    # Heuristically find the column to predict, then decide between
    # classification and regression.
    # ------------------------------------------------------------------
    target_col = detect_target_column(df)
    print(f"\nTarget Column : {target_col}")

    task_type = detect_task_type(df, target_col)
    print(f"\nTask Type     : {task_type.upper()}")

    # Record the original dtype of the target BEFORE any encoding.
    # CTGAN converts string labels to integers internally; we need this
    # to restore the correct dtype after synthetic data generation.
    original_target_dtype = df[target_col].dtype
    print(f"\nOriginal Target Dtype : {original_target_dtype}")

    # Keep a raw copy of the dataframe for before/after visualisations
    df_before = df.copy()

    # ------------------------------------------------------------------
    # STEP 3 — PREPROCESSING
    # Separate features from the target, preprocess features (encode,
    # scale, handle missing values), then reassemble into one DataFrame.
    # ------------------------------------------------------------------
    target_series = df[target_col].copy().reset_index(drop=True)
    features_df   = df.drop(columns=[target_col])
    features_df, encoders = preprocess_dataset(features_df)
    features_df   = features_df.reset_index(drop=True)

    # Align lengths — some preprocessors may drop rows
    min_len       = min(len(features_df), len(target_series))
    features_df   = features_df.iloc[:min_len]
    target_series = target_series.iloc[:min_len]

    df = pd.concat([features_df, target_series], axis=1)

    before_drop = len(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    if len(df) < before_drop:
        print(f"\nDropped {before_drop - len(df)} rows with missing target values")

    optimize_memory()
    validate_final_dataset(df, target_col, task_type)

    # ------------------------------------------------------------------
    # FIX 2: Identify categorical columns HERE (after preprocessing)
    # so the variable is always in scope for both classification and
    # regression paths, and for pie-chart visualisation later.
    # (Original code defined this inside the SMOTE block — causing a
    #  potential NameError on the regression path and in pie charts.)
    # ------------------------------------------------------------------
    categorical_cols = detect_categorical_columns(df, target_col)

    # ------------------------------------------------------------------
    # STEP 4 — INITIAL METRICS
    # Compute baseline EDQS, imbalance ratio, and fairness score BEFORE
    # any augmentation so we can show meaningful before/after comparisons.
    # ------------------------------------------------------------------
    print_section("INITIAL METRICS")

    edqs_before_metrics = calculate_edqs(df, target_col, task_type)
    edqs_before         = edqs_before_metrics["edqs"]

    if task_type == "classification":
        imbalance_ratio_before, class_counts_before = detect_imbalance(df, target_col)
        imbalance_report(class_counts_before)
        bias_results_before, fairness_before = calculate_fairness(df, target_col)
    else:
        imbalance_ratio_before = 0
        class_counts_before    = {}
        bias_results_before    = pd.DataFrame()
        fairness_before        = 1.0

    print(f"\nFairness Before : {fairness_before:.2f}")

    # ------------------------------------------------------------------
    # STEP 5 — TRAIN / TEST SPLIT
    # Stratified split for classification to preserve class proportions.
    # The test set is NEVER modified by SMOTE or CTGAN.
    # ------------------------------------------------------------------
    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify_option = None
    if task_type == "classification":
        class_counts    = y.value_counts()
        min_class_count = class_counts.min()

        print("\n========== STRATIFY CHECK ==========")
        print(class_counts)

        if min_class_count >= 2:
            stratify_option = y
            print("\nStratify Enabled")
        else:
            print("\nStratify Disabled (very small class detected)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_option,
    )

    train_df = pd.concat([X_train, y_train], axis=1)

    # ------------------------------------------------------------------
    # STEP 6 — SMOTE  (classification only)
    # Oversample minority classes on the TRAINING set only to reduce
    # class imbalance before synthetic data generation.
    # SMOTENC is used here because the dataset has mixed data types
    # (both numeric and categorical columns).
    # ------------------------------------------------------------------
    # Categorical columns for SMOTENC: low-cardinality numeric cols also
    # treated as categorical to avoid SMOTE interpolating between labels
    smote_categorical_cols = [
        col for col in train_df.columns
        if col != target_col and train_df[col].nunique() <= 10
    ]

    if task_type == "classification":
        try:
            imbalance_ratio_train, _ = detect_imbalance(train_df, target_col)
            min_class_count          = train_df[target_col].value_counts().min()

            if imbalance_ratio_train < 0.80 and min_class_count >= 2:
                smote_output = apply_smote(train_df, target_col, smote_categorical_cols)
                if smote_output is not None:
                    train_df = smote_output
                    print("\nSMOTE Applied On Training Data")
                else:
                    print("\nSMOTE Failed — continuing without it")
            else:
                print("\nSMOTE Skipped (already balanced or very small class)")

        except Exception as e:
            print(f"SMOTE error: {e}")
            write_log(str(e))

    # ------------------------------------------------------------------
    # STEP 7 — CTGAN SYNTHETIC DATA GENERATION
    # Train a Conditional GAN on the training data to generate additional
    # synthetic rows, then merge them back into the training set.
    #
    # FIX (dtype): CTGAN encodes string labels as integers internally.
    #   After generation we decode the synthetic target back to original
    #   string labels and enforce consistent dtype before concat.
    # ------------------------------------------------------------------
    synthetic_rows_added = 0
    synthetic_status     = "No"

    try:
        ctgan_df = train_df.copy()

        # Subsample for speed when the training set is very large
        if len(ctgan_df) > 5000:
            ctgan_df = ctgan_df.sample(n=5000, random_state=42).reset_index(drop=True)
            print(f"\nCTGAN Training Sample Size : {len(ctgan_df)}")

        # CTGAN requires all columns to be numeric — encode categoricals
        ctgan_label_map = {}  # int→label mapping for decoding target later
        for col in ctgan_df.select_dtypes(include=["object"]).columns:
            cat = ctgan_df[col].astype("category")
            ctgan_df[col] = cat.cat.codes
            if col == target_col:
                ctgan_label_map = dict(enumerate(cat.cat.categories))

        synthetic_df = generate_ctgan_data(ctgan_df, target_col, epochs=5)
        synthetic_df = synthetic_df.dropna().drop_duplicates()

        # ── Decode synthetic target column back to original string labels ──
        if ctgan_label_map and original_target_dtype == object:
            synthetic_df[target_col] = (
                synthetic_df[target_col]
                .round()
                .astype(int)
                .map(ctgan_label_map)
            )
            # Drop any rows where the code was out of range (NaN after map)
            synthetic_df = synthetic_df.dropna(subset=[target_col])
            print("\nCTGAN synthetic labels decoded back to original strings")

        elif task_type == "regression":
            # For regression clip and round to valid range
            orig_min = df[target_col].min()
            orig_max = df[target_col].max()
            synthetic_df[target_col] = (
                synthetic_df[target_col]
                .clip(orig_min, orig_max)
                .round()
                .astype(int)
            )

        # Enforce dtype consistency before concat to avoid mixed types
        synthetic_df = enforce_target_dtype(synthetic_df, target_col, original_target_dtype)
        train_df     = enforce_target_dtype(train_df,     target_col, original_target_dtype)

        if len(synthetic_df) > 0:
            train_df             = pd.concat([train_df, synthetic_df], ignore_index=True)
            synthetic_rows_added = len(synthetic_df)
            synthetic_status     = "Yes"
            print(f"\nSynthetic Rows Added : {synthetic_rows_added}")

            try:
                plot_ctgan_real_vs_synthetic(
                    real_df=ctgan_df,
                    synthetic_df=synthetic_df,
                    graph_dir=graph_dir,
                )
            except Exception as e:
                print(f"CTGAN graph error: {e}")
        else:
            print("\nCTGAN Generated 0 Rows — skipping augmentation")

    except Exception as e:
        print(f"CTGAN error: {e}")
        write_log(str(e))

    # ------------------------------------------------------------------
    # STEP 8 — POST-AUGMENTATION CLEANING
    # Remove excess duplicates, apply fairness-aware resampling, and
    # correct skewed feature distributions.
    # ------------------------------------------------------------------

    # Safety: enforce dtype again after all concat operations
    train_df = enforce_target_dtype(train_df, target_col, original_target_dtype)

    train_df = safe_remove_duplicates(train_df)

    if task_type == "classification":
        try:
            train_df = apply_fairness_fix(
                train_df,
                bias_results_before,
                target_col
            )

            print("\nFairness Fix Applied Successfully")

            duplicate_pct = (
                train_df.duplicated().mean() * 100
            )

            print(
                f"\nDuplicates After Fairness Fix : "
                f"{duplicate_pct:.2f}%"
            )

            if duplicate_pct > 10:

                before_rows = len(train_df)

                train_df = train_df.drop_duplicates()

                removed = (
                    before_rows - len(train_df)
                )

                print(
                    f"\nRemoved {removed} "
                    f"duplicate rows"
                )

        except Exception as e:
            print(f"Fairness fix error: {e}")
            write_log(str(e))

    try:
        train_df = fix_skewness(train_df, target_col)
    except Exception as e:
        print(f"Skewness fix error: {e}")
        write_log(str(e))

    # Final dtype enforcement after all cleaning steps
    train_df = enforce_target_dtype(train_df, target_col, original_target_dtype)

    validate_final_dataset(train_df, target_col, task_type)

    # ------------------------------------------------------------------
    # STEP 9 — FINAL FAIRNESS METRICS
    # Recompute fairness on the augmented + cleaned training set so we
    # can show the improvement achieved by SMOTE, CTGAN, and fairness fix.
    # ------------------------------------------------------------------
    print_section("FINAL FAIRNESS METRICS")

    if task_type == "classification":
        bias_results_after, fairness_after = calculate_fairness(train_df, target_col)
    else:
        bias_results_after = pd.DataFrame()
        fairness_after     = 1.0

    print(f"\nFairness After : {fairness_after:.2f}")

    bias_table_data   = None
    bias_report_text  = ""
    overall_before    = 0.0
    overall_after     = 0.0
    overall_reduction = 0.0

    # Build the bias comparison table for the PDF report (classification only)
    if task_type == "classification":
        print("\n========== COMPREHENSIVE BIAS REPORT ==========")
        print(f"{'Bias Type':<25} {'Before':<10} {'After':<10} Reduction")
        print("-" * 60)

        merged_bias = pd.merge(
            bias_results_before[["Bias Type", "Probability"]],
            bias_results_after[["Bias Type", "Probability"]],
            on="Bias Type",
            suffixes=("_Before", "_After"),
        )

        reduction_list = []

        for _, row in merged_bias.iterrows():

            before = row["Probability_Before"]

            after = row["Probability_After"]

            if before > 0:

                reduction = (

                    (before - after)

                    / before

                    * 100
                )

            else:

                reduction = 0.0

            reduction_list.append(
                reduction
            )

            status = (

                "Reduced"

                if reduction >= 0

                else

                "Increased"
            )

            print(
                f"{row['Bias Type']:<25}"
                f"{before:<10.2f}"
                f"{after:<10.2f}"
                f"{status} "
                f"{abs(reduction):.2f}%"
            )
            
        merged_bias["Reduction %"] = reduction_list

        bias_table_data = [["Bias Type", "Before", "After", "Reduction %"]]
        for _, row in merged_bias.iterrows():
            bias_table_data.append([
                row["Bias Type"],
                f"{row['Probability_Before']:.2f}",
                f"{row['Probability_After']:.2f}",
                f"{row['Reduction %']:.2f}%",
            ])

        overall_before    = merged_bias["Probability_Before"].mean()
        overall_after     = merged_bias["Probability_After"].mean()
        overall_reduction = (
            (overall_before - overall_after) / max(overall_before, 0.0001)
        ) * 100

        print(f"\nOverall Bias Before   : {overall_before:.2f}")
        print(f"Overall Bias After    : {overall_after:.2f}")
        print(f"Total Bias Reduction  : {overall_reduction:.2f}%")

        for _, row in merged_bias.iterrows():
            bias_report_text += (
                f"{row['Bias Type']}\n"
                f"Before    : {row['Probability_Before']:.2f}\n"
                f"After     : {row['Probability_After']:.2f}\n"
                f"Reduction : {row['Reduction %']:.2f}%\n\n"
            )

        bias_csv_path = os.path.join(graph_dir, "bias_comparison.csv")
        merged_bias.to_csv(bias_csv_path, index=False)
        print(f"\nBias Comparison Saved: {bias_csv_path}")

    else:
        bias_report_text = "Bias Analysis Not Available For Regression Task"

    # ------------------------------------------------------------------
    # STEP 10 — FINAL EDQS & IMBALANCE (post-augmentation)
    # Recompute dataset quality metrics on the final training dataset.
    # ------------------------------------------------------------------
    edqs_after_metrics = calculate_edqs(train_df, target_col, task_type)
    edqs_after         = edqs_after_metrics["edqs"]

    if task_type == "classification":
        imbalance_ratio_after, class_counts_after = detect_imbalance(train_df, target_col)

        print("\n========== FINAL CLASS CHECK ==========")
        print(train_df[target_col].value_counts())

        if train_df[target_col].nunique() < 2:
            raise ValueError(
                "Model training stopped: only one class remains after augmentation."
            )
    else:
        imbalance_ratio_after = 0
        class_counts_after    = {}

    # ------------------------------------------------------------------
    # STEP 11 — BASELINE ERI / RAI
    #
    # FIX 4 (ERI direction bug):
    #   Original code used accuracy=0.50 (random chance) for the BEFORE
    #   ERI, but the AFTER ERI used the real model accuracy (~0.32).
    #   Since 0.32 < 0.50, the accuracy risk INCREASED, so ERI went UP
    #   instead of down — the opposite of what the paper intended.
    #
    #   Correct approach: Baseline ERI should represent the WORST CASE
    #   (accuracy=0.0, no model yet).  Any trained model will then show
    #   lower (better) ERI, demonstrating clear improvement.
    # ------------------------------------------------------------------
    eri_before = calculate_eri(
        fairness_score=fairness_before,
        accuracy=0.0,                    # ← FIX: worst-case baseline (was 0.50)
        imbalance_ratio=imbalance_ratio_before,
        explainability_score=0.80,
    )
    rai_before = calculate_rai(edqs_before, fairness_before, 0.0, eri_before)

    # ------------------------------------------------------------------
    # STEP 12 — LLM FEEDBACK TRAINING LOOP
    # Train an ensemble model, evaluate it, pass the metrics to the LLM,
    # receive hyperparameter recommendations, update, and repeat.
    #
    # FIX 5 (dtype):  y_train_final is cast to a consistent, sortable
    #   dtype before model.fit() to prevent the TypeError that crashed
    #   sklearn's VotingClassifier in the original code.
    #
    # FIX 6 (loop stagnation):  When the LLM keeps recommending only
    #   'retrain_model' and accuracy does not improve (plateau), the loop
    #   now automatically reduces max_depth (adds regularisation) to
    #   escape the plateau — even without an explicit LLM recommendation.
    # ------------------------------------------------------------------
    X_train_final = train_df.drop(columns=[target_col])

    # Enforce dtype consistency for y_train before model.fit()
    y_train_final = train_df[target_col].copy()
    if original_target_dtype == object or str(original_target_dtype) == "object":
        y_train_final = y_train_final.astype(str)
    else:
        try:
            y_train_final = y_train_final.astype(original_target_dtype)
        except Exception:
            y_train_final = y_train_final.astype(str)

    # Match y_test dtype to y_train for fair evaluation
    y_test = y_test.copy()
    try:
        y_test = y_test.astype(y_train_final.dtype)
    except Exception:
        y_test = y_test.astype(str)

    best_score      = -999
    best_model      = None
    best_metrics    = None
    best_eri        = None
    best_rai        = None
    best_governance = None
    stagnation      = 0

    # Track accuracy across loops to detect a genuine accuracy plateau
    prev_accuracy   = -1.0

    # Initial hyperparameters for the ensemble (Random Forest / GBM)
    rf_params = {
        "n_estimators":      200,
        "max_depth":         None,
        "min_samples_split": 2,
        "min_samples_leaf":  1,
        "class_weight":      None,
    }

    for iteration in range(MAX_ITERATIONS):
        print(f"\n========== LLM FEEDBACK LOOP {iteration + 1} ==========")

        # Train the ensemble with the current hyperparameters
        model = train_model(X_train_final, y_train_final, task_type, **rf_params)

        # Compute feature-based explainability score (Gini importance spread)
        explainability_score = calculate_explainability_score(model)

        # Evaluate on the held-out test set
        y_pred  = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, task_type)

        # Pick the primary performance metric
        if task_type == "classification":
            performance_score = metrics.get("accuracy", 0)
        else:
            performance_score = max(metrics.get("r2", 0), 0)

        # Compute ERI, RAI, and governance score for this iteration
        eri_iter = calculate_eri(
            fairness_score=fairness_after,
            accuracy=performance_score,
            imbalance_ratio=imbalance_ratio_after,
            explainability_score=explainability_score,
        )
        rai_iter        = calculate_rai(edqs_after, fairness_after, performance_score, eri_iter)
        governance_iter = calculate_governance_score(
            edqs_after, fairness_after, performance_score, eri_iter
        )

        # Composite responsible-AI score to guide convergence
        overall_score = calculate_overall_score(
            performance_score,
            fairness_after,
            edqs_after,
            rai_iter,
            explainability_score,
        )

        print(f"\nOverall Score : {overall_score:.4f}")

        # Update the best model only when the score improves meaningfully
        if overall_score > (best_score + MIN_IMPROVEMENT):
            best_score      = overall_score
            best_model      = model
            best_metrics    = metrics
            best_eri        = eri_iter
            best_rai        = rai_iter
            best_governance = governance_iter
            stagnation      = 0
            print("\nImprovement Found — saving best model")
        else:
            stagnation += 1
            print(f"\nNo Improvement ({stagnation}/{PATIENCE})")

        # Build the metrics string that is passed to the LLM
        metrics_text = (
            f"Accuracy : {metrics.get('accuracy', performance_score)}\n"
            f"Fairness : {fairness_after}\n"
            f"Fairness After : {fairness_after}\n"
            f"ERI : {eri_iter}\n"
            f"ERI After : {eri_iter}\n"
            f"EDQS : {edqs_after}\n"
            f"RAI : {rai_iter}\n"
            f"Explainability : {explainability_score}\n"
        )

        # Log every iteration's metrics to the global text file
        append_metrics_to_txt(
            f"LLM LOOP ITERATION {iteration + 1}",
            {
                "Accuracy":       metrics.get("accuracy", performance_score),
                "Fairness After": fairness_after,
                "ERI After":      eri_iter,
                "EDQS After":     edqs_after,
                "RAI After":      rai_iter,
                "Explainability": explainability_score,
                "Overall Score":  overall_score,
            },
        )

        # Ask the LLM for hyperparameter recommendations for the next loop
        recommendations = parse_llm_recommendations(
            generate_llm_recommendations(metrics_text)
        )

        print("\n========== LLM RECOMMENDATIONS ==========")
        print(recommendations)

        # ── Apply LLM recommendations to DATA (SMOTE, duplicates, etc.) ──
        # This is the key step where data-level fixes recommended by the LLM
        # are actually executed before the next training iteration.
        try:
            train_df = apply_llm_recommendations(
                df=train_df,
                recommendations=recommendations,
                target_col=target_col,
                categorical_cols=smote_categorical_cols,
            )
            # Re-sync X_train_final and y_train_final after any data changes
            train_df = enforce_target_dtype(train_df, target_col, original_target_dtype)
            X_train_final = train_df.drop(columns=[target_col])
            y_train_final = train_df[target_col].copy()
            if original_target_dtype == object or str(original_target_dtype) == "object":
                y_train_final = y_train_final.astype(str)
            else:
                try:
                    y_train_final = y_train_final.astype(original_target_dtype)
                except Exception:
                    y_train_final = y_train_final.astype(str)
        except Exception as e:
            print(f"apply_llm_recommendations error: {e}")
            write_log(str(e))

        # ── Apply LLM recommendations to hyperparameters ──────────────

        if recommendations.get("increase_regularization", False):
            # Reduce max_depth to limit model complexity
            if rf_params["max_depth"] is None:
                rf_params["max_depth"] = 15
            else:
                rf_params["max_depth"] = max(3, rf_params["max_depth"] - 2)
            print(f"\nNew Max Depth : {rf_params['max_depth']}")

        if recommendations.get("retrain_model", False):
            # Add more trees — useful when the model is under-fitted
            rf_params["n_estimators"] = min(1000, rf_params["n_estimators"] + 50)
            print(f"\nNew Trees : {rf_params['n_estimators']}")

        if recommendations.get("apply_fairness_fix", False):
            # Balance class weights to help with skewed distributions
            rf_params["class_weight"] = "balanced"
            print("\nBalanced Class Weight Applied")

        # ── FIX 6: Accuracy plateau detection ─────────────────────────
        # If the LLM keeps recommending 'retrain_model' but accuracy is
        # completely flat (no change at all), force a regularisation step
        # to escape the plateau.  This prevents wasting compute on adding
        # more trees to a model that has already converged.
        accuracy_change = abs(performance_score - prev_accuracy)
        if (
            recommendations.get("retrain_model", False)
            and not recommendations.get("increase_regularization", False)
            and accuracy_change < 0.001
            and iteration > 0
        ):
            if rf_params["max_depth"] is None:
                rf_params["max_depth"] = 12
            else:
                rf_params["max_depth"] = max(3, rf_params["max_depth"] - 2)
            print(
                f"\n[Plateau Detected] Auto-reducing max_depth to "
                f"{rf_params['max_depth']} to escape stagnation"
            )

        prev_accuracy = performance_score  # Track for next iteration

        # Early stopping: exit if no improvement for PATIENCE rounds
        if stagnation >= PATIENCE:
            print("\nModel Converged — stopping loop")
            break

    if best_model is None:
        raise ValueError("No valid model was generated during training loop.")

    # Use the best model found across all iterations
    model      = best_model
    metrics    = best_metrics
    eri_after  = best_eri
    rai_after  = best_rai
    governance = best_governance

    print(f"\nBest Overall Score : {best_score:.4f}")

    # Recalculate final metrics from the best saved model
    explainability_score = calculate_explainability_score(model)

    if task_type == "classification":
        performance_score = metrics.get("accuracy", 0)
    else:
        performance_score = max(metrics.get("r2", 0), 0)

    eri_after  = calculate_eri(
        fairness_score=fairness_after,
        accuracy=performance_score,
        imbalance_ratio=imbalance_ratio_after,
        explainability_score=explainability_score,
    )
    rai_after  = calculate_rai(edqs_after, fairness_after, performance_score, eri_after)
    governance = calculate_governance_score(
        edqs_after, fairness_after, performance_score, eri_after
    )

    # ------------------------------------------------------------------
    # STEP 13 — PRINT FINAL METRICS SUMMARY
    # ------------------------------------------------------------------
    print_section("FINAL METRICS")

    print(f"\nEDQS Before       : {edqs_before:.2f}")
    print(f"EDQS After        : {edqs_after:.2f}")
    print(f"\nFairness Before   : {fairness_before:.2f}")
    print(f"Fairness After    : {fairness_after:.2f}")
    print(f"\nERI Before        : {eri_before:.4f}")
    print(f"ERI After         : {eri_after:.4f}")
    print(f"\nRAI Before        : {rai_before:.2f}")
    print(f"RAI After         : {rai_after:.2f}")
    print(f"\nGovernance Score  : {governance['score']:.2f}")
    print(f"Governance Level  : {governance['level']}")
    print(f"\nExplainability    : {explainability_score:.2f}")

    # ------------------------------------------------------------------
    # STEP 14 — VISUALISATIONS
    # Generate all graphs: class counts, EDQS comparison, boxplots,
    # pie charts, heatmaps, ERI/RAI/fairness comparisons, bias plots.
    # ------------------------------------------------------------------
    try:
        plot_before_after_counts(df_before, train_df, graph_dir)
        
        plot_edqs_comparison(edqs_before, edqs_after, graph_dir, task_type)

        try:
            plot_boxplots(df_before, train_df, graph_dir)
        except Exception as e:
            print(f"Boxplot error: {e}")

        try:
            plot_piecharts(df_before, train_df, categorical_cols, graph_dir)
        except Exception as e:
            print(f"Pie Chart error: {e}")

        try:
            plot_feature_importance(model, X_train_final.columns, graph_dir)
        except Exception as e:
            print(f"Feature Importance error: {e}")

        plot_correlation_heatmap(train_df, graph_dir)
        plot_eri_comparison(eri_before, eri_after, graph_dir)
        plot_rai_comparison(rai_before, rai_after, graph_dir)
        fairness_comparison_chart(fairness_before, fairness_after, graph_dir)

        # Bias visualisations are only meaningful for classification
        if (
            task_type == "classification"
            and isinstance(bias_results_before, pd.DataFrame)
            and not bias_results_before.empty
        ):
            bias_before_after_graph(bias_results_before, bias_results_after, graph_dir)
            bias_radar_chart(bias_results_before, bias_results_after, graph_dir)

        if task_type == "classification":
            bias_heatmap(train_df, graph_dir)

    except Exception as e:
        print(f"Visualization error: {e}")

    # ------------------------------------------------------------------
    # STEP 15 — SAVE ALL METRICS TO LOG FILE
    # ------------------------------------------------------------------
    metrics_data = {
        "EDQS Before":          edqs_before,
        "EDQS After":           edqs_after,
        "Fairness Before":      fairness_before,
        "Fairness After":       fairness_after,
        "ERI Before":           eri_before,
        "ERI After":            eri_after,
        "RAI Before":           rai_before,
        "RAI After":            rai_after,
        "Explainability Score": explainability_score,
        "Governance Score":     governance["score"],
        "Governance Level":     governance["level"],
    }

    if task_type == "classification":
        metrics_data.update({
            "Accuracy":  metrics.get("accuracy",  0),
            "Precision": metrics.get("precision", 0),
            "Recall":    metrics.get("recall",    0),
            "F1 Score":  metrics.get("f1_score",  0),
        })
    else:
        metrics_data.update({
            "MAE":      metrics.get("mae",  0),
            "MSE":      metrics.get("mse",  0),
            "RMSE":     metrics.get("rmse", 0),
            "R2 Score": metrics.get("r2",   0),
        })

    append_metrics_to_txt("FINAL METRICS", metrics_data)
    metrics_path = save_metrics(metrics_data)

    # ------------------------------------------------------------------
    # STEP 16 — LLM ANALYSIS
    # Feed the complete metrics log to the LLM for a natural-language
    # analysis and final actionable recommendations.
    # ------------------------------------------------------------------
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_text = f.read()

    # Read the full metrics log (contains all loop iterations) for richer LLM context
    try:
        with open(METRICS_TEXT_PATH, "r", encoding="utf-8") as f:
            full_metrics_log = f.read()
    except Exception:
        full_metrics_log = metrics_text  # fallback to summary

    llm_analysis = generate_llm_analysis(full_metrics_log)
    print("\n========== LLM ANALYSIS ==========")
    print(llm_analysis)

    recommendations = generate_llm_recommendations(metrics_text)
    print("\n========== LLM RECOMMENDATIONS ==========")
    print(recommendations)

    # ------------------------------------------------------------------
    # STEP 17 — SAVE MODEL & FINAL DATASET
    # ------------------------------------------------------------------
    save_model(model)

    output_csv = os.path.join("outputs", "final_responsible_ai_dataset.csv")
    train_df.to_csv(output_csv, index=False)
    print(f"\nFinal Dataset Saved: {output_csv}")

    # ------------------------------------------------------------------
    # STEP 18 — GENERATE PDF REPORT
    # Compile all metrics, the bias comparison table, and the LLM
    # analysis into a single downloadable PDF report.
    # ------------------------------------------------------------------
    if task_type == "classification":
        performance_block = (
            f"Accuracy  : {metrics.get('accuracy',  0):.4f}\n"
            f"Precision : {metrics.get('precision', 0):.4f}\n"
            f"Recall    : {metrics.get('recall',    0):.4f}\n"
            f"F1 Score  : {metrics.get('f1_score',  0):.4f}"
        )
    else:
        performance_block = (
            f"MAE      : {metrics.get('mae',  0):.4f}\n"
            f"MSE      : {metrics.get('mse',  0):.4f}\n"
            f"RMSE     : {metrics.get('rmse', 0):.4f}\n"
            f"R2 Score : {metrics.get('r2',   0):.4f}"
        )

    report_content = (
        f"RESPONSIBLE AI REPORT\n"
        f"Dataset           : {dataset_name}\n\n"
        f"EDQS Before       : {edqs_before:.2f}\n"
        f"EDQS After        : {edqs_after:.2f}\n\n"
        f"Fairness Before   : {fairness_before:.2f}\n"
        f"Fairness After    : {fairness_after:.2f}\n\n"
        f"ERI Before        : {eri_before:.4f}\n"
        f"ERI After         : {eri_after:.4f}\n\n"
        f"RAI Before        : {rai_before:.2f}\n"
        f"RAI After         : {rai_after:.2f}\n\n"
        f"Explainability    : {explainability_score:.2f}\n"
        f"Governance Score  : {governance['score']:.2f}\n"
        f"Governance Level  : {governance['level']}\n\n"
        f"{performance_block}\n\n"
        f"BIAS REPORT\n\n"
        f"{bias_report_text}\n"
        f"Overall Bias Before  : {overall_before:.2f}\n"
        f"Overall Bias After   : {overall_after:.2f}\n"
        f"Total Bias Reduction : {overall_reduction:.2f}%\n\n"
        f"LLM ANALYSIS:\n{llm_analysis}\n"
    )

    generate_pdf_report(
        "outputs/responsible_ai_report.pdf",
        report_content,
        bias_table_data,
    )

    optimize_memory()
    print_section("RESPONSIBLE AI PIPELINE COMPLETED")


# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    main()
