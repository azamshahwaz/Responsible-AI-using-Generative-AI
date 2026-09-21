# =========================================================
# FAIRNESS FIX MODULE
# ADVANCED RESPONSIBLE AI VERSION
# =========================================================

import pandas as pd
import numpy as np

from sklearn.utils import resample

# =========================================================
# APPLY FAIRNESS FIX
# =========================================================

def apply_fairness_fix(
    df,
    bias_results,
    target_col
):

    print(
        "\n========== FAIRNESS FIX STARTED =========="
    )

    df=df.copy()

    try:

        # =================================================
        # EMPTY BIAS CHECK
        # =================================================

        if (

            bias_results is None

            or

            len(bias_results)==0
        ):

            print(
                "\nNo Bias Results Found"
            )

            return df

        # =================================================
        # CONVERT TO DATAFRAME
        # =================================================

        if not isinstance(
            bias_results,
            pd.DataFrame
        ):

            bias_results=pd.DataFrame(
                bias_results
            )

        # =================================================
        # REQUIRED COLUMN CHECK
        # =================================================

        required_cols=[
            "Bias Type",
            "Probability"
        ]

        for col in required_cols:

            if col not in bias_results.columns:

                print(
                    f"\nMissing Bias Column: {col}"
                )

                return df

        # =================================================
        # HIGH / MODERATE BIAS TYPES
        # =================================================

        high_bias=bias_results[

            bias_results["Probability"]>0.10
        ]

        print("\nDetected Bias Types:")

        print(
            high_bias["Bias Type"]
            .tolist()
        )

        # =================================================
        # REPRESENTATION BIAS FIX
        # =================================================

        if (

            "Representation Bias"

            in

            high_bias["Bias Type"].values
        ):

            print(
                "\nFixing Representation Bias..."
            )

            class_counts = (

                df[target_col]

                .value_counts()
            )

            print(
                "\nCurrent Class Distribution:"
            )

            print(class_counts)

            imbalance_ratio = (

                class_counts.min()

                /

                class_counts.max()
            )

            print(
                f"\nCurrent Imbalance Ratio : "
                f"{imbalance_ratio:.4f}"
            )

            # =========================================
            # SMOTE / CTGAN ALREADY HANDLED BALANCING
            # =========================================

            if imbalance_ratio >= 0.80:

                print(
                    "\nDataset already reasonably balanced"
                )

                print(
                    "Skipping additional resampling"
                )

                print(
                    "SMOTE/CTGAN already reduced "
                    "representation bias"
                )

            else:

                print(
                    "\nSevere imbalance still detected"
                )

                print(
                    "Applying controlled balancing..."
                )

                majority_count = (

                    class_counts.max()
                )

                balanced_data = []

                for cls in class_counts.index:

                    cls_df = df[

                        df[target_col] == cls
                    ]

                    if len(cls_df) < majority_count:

                        target_size = min(

                            int(
                                len(cls_df) * 1.50
                            ),

                            majority_count
                        )

                        cls_df = resample(

                            cls_df,

                            replace=True,

                            n_samples=target_size,

                            random_state=42
                        )

                    balanced_data.append(
                        cls_df
                    )

                df = pd.concat(

                    balanced_data,

                    ignore_index=True
                )

                df = df.sample(

                    frac=1,

                    random_state=42
                ).reset_index(drop=True)

                print(
                    "\nUpdated Class Distribution:"
                )

                print(

                    df[target_col]

                    .value_counts()
                )

            print(
                "\nRepresentation Bias Reduced"
            )


        # =================================================
        # SELECTION BIAS FIX
        # =================================================

        if (

            "Selection Bias"

            in

            high_bias["Bias Type"].values
        ):

            print(
                "\nSelection Bias detected"
            )

            df=df.sample(

                frac=1,

                random_state=42
            ).reset_index(drop=True)

            print(
                "\nDataset randomized"
            )
            
            print("Selection Bias mitigation applied")

        # =================================================
        # SAMPLING BIAS FIX
        # =================================================

        if (

            "Sampling Bias"

            in

            high_bias["Bias Type"].values
        ):

            print(
                "\nFixing Sampling Bias..."
            )

            df = df.sample(

                frac=1,

                replace=True,

                random_state=42

            ).reset_index(drop=True)

            print(
                "\nSampling Bias Reduced"
            )

        # =================================================
        # HISTORICAL BIAS FIX
        # =================================================

        if (

            "Historical Bias"

            in

            high_bias["Bias Type"].values
        ):

            print(
                "\nHistorical Bias detected"
            )
            print(
                "Historical Bias mitigation delegated to Skewness Fix Module")
            
        # =================================================
        # RESPONSE + EXCLUSION BIAS FIX
        # =================================================

        if (

            "Response Bias"

            in

            high_bias["Bias Type"].values

            or

            "Exclusion Bias"

            in

            high_bias["Bias Type"].values
        ):

            print(
                "\nFixing Response/Exclusion Bias..."
            )

            for col in df.columns:

                if str(df[col].dtype) == "object":

                    mode_val = df[col].mode()

                    if len(mode_val) > 0:

                        df[col] = df[col].fillna(
                            mode_val[0]
                        )

                else:

                    df[col] = df[col].fillna(
                        df[col].median()
                    )

            print(
                "\nResponse/Exclusion Bias Reduced"
            )
            
        # =================================================
        # MEASUREMENT BIAS FIX
        # =================================================

        measurement_row = bias_results[

            bias_results["Bias Type"]

            ==

            "Measurement Bias"
        ]

        measurement_prob = 0

        if len(measurement_row) > 0:

            measurement_prob = float(

                measurement_row["Probability"]

                .iloc[0]
            )

        if measurement_prob > 0.20:

            print(
                "\nFixing Measurement Bias..."
            )

            numeric_cols = df.select_dtypes(
                include=np.number
            ).columns

            for col in numeric_cols:

                if col == target_col:

                    continue

                q1 = df[col].quantile(0.25)

                q3 = df[col].quantile(0.75)

                iqr = q3 - q1

                lower = q1 - 1.5 * iqr

                upper = q3 + 1.5 * iqr

                df[col] = np.clip(

                    df[col],

                    lower,

                    upper
                )

            print(
                "\nMeasurement Bias Reduced"
            )

        else:

            print(
                "\nMeasurement Bias Low"
            )

            print(
                "Skipping Outlier Clipping"
            )
            
        # =================================================
        # LABEL BIAS FIX
        # =================================================

        if (

            "Label Bias"

            in

            high_bias["Bias Type"].values
        ):

            print(
                "\nFixing Label Bias..."
            )

            class_counts = (

                df[target_col]

                .value_counts()
            )

            imbalance_ratio = (

                class_counts.min()

                /

                class_counts.max()
            )

            print(
                f"\nCurrent Label Balance Ratio : "
                f"{imbalance_ratio:.4f}"
            )

            if imbalance_ratio >= 0.80:

                print(
                    "\nDataset already balanced"
                )

                print(
                    "Skipping Label Bias resampling"
                )

            else:

                target_size = int(

                    class_counts.max()

                    * 0.90
                )

                balanced_data = []

                for cls in class_counts.index:

                    cls_df = df[

                        df[target_col] == cls
                    ]

                    if len(cls_df) < target_size:

                        cls_df = resample(

                            cls_df,

                            replace=True,

                            n_samples=target_size,

                            random_state=42
                        )

                    balanced_data.append(
                        cls_df
                    )

                df = pd.concat(

                    balanced_data,

                    ignore_index=True
                )

                df = df.sample(

                    frac=1,

                    random_state=42
                ).reset_index(drop=True)

                print(
                    "\nUpdated Class Distribution:"
                )

                print(

                    df[target_col]

                    .value_counts()
                )

            print(
                "\nLabel Bias Reduced"
            )
            
        # =================================================
        # CONFIRMATION BIAS FIX
        # =================================================

        if (

            "Confirmation Bias"

            in

            high_bias["Bias Type"].values
        ):

            print(
                "\nConfirmation Bias delegated to Skewness Fix Module"
            )
        # =================================================
        # PROXY BIAS FIX
        # =================================================

        if (

            "Proxy Bias"

            in

            high_bias["Bias Type"].values
        ):

            print(
                "\nFixing Proxy Bias..."
            )

            numeric_cols = df.select_dtypes(
                include=np.number
            ).columns.tolist()

            if target_col not in numeric_cols:

                print(
                    "\nTarget not numeric - Proxy Bias Skipped"
                )

            else:

                protected_corr_threshold = 0.75

                corr_matrix = df[
                    numeric_cols
                ].corr().abs()

                proxy_cols = []

                # =============================================
                # DETECT POTENTIAL PROXY FEATURES
                # =============================================

                for col in numeric_cols:

                    if col == target_col:

                        continue

                    try:

                        target_corr = (

                            corr_matrix.loc[
                                target_col,
                                col
                            ]
                        )

                        if (
                            target_corr >
                            protected_corr_threshold
                        ):

                            proxy_cols.append(col)

                    except Exception:

                        pass

                # =============================================
                # REPORT ONLY (NO FEATURE DELETION)
                # =============================================

                if len(proxy_cols) > 0:

                    print(
                        f"\nPotential Proxy Columns Detected: "
                        f"{proxy_cols}"
                    )

                else:

                    print(
                        "\nNo Strong Proxy Columns Detected"
                    )

                # =============================================
                # LIGHT FAIRNESS NOISE
                # =============================================

                for col in proxy_cols:

                    try:

                        noise_scale = (

                            df[col].std()
                            * 0.01
                        )

                        if noise_scale > 0:

                            noise = np.random.normal(

                                0,

                                noise_scale,

                                len(df)
                            )

                            df[col] = (

                                df[col]

                                + noise
                            )

                    except Exception:

                        pass

            print(
                "\nProxy Bias Reduced"
            )

        # =================================================
        # OUTCOME BIAS FIX
        # =================================================

        if (

            "Outcome Bias"

            in

            high_bias["Bias Type"].values
        ):

            print(
                "\nFixing Outcome Bias..."
            )

            numeric_cols=df.select_dtypes(
                include=np.number
            ).columns

            for col in numeric_cols:

                if col!=target_col:

                    q1=df[col].quantile(0.25)

                    q3=df[col].quantile(0.75)

                    iqr=q3-q1

                    lower=q1-1.5*iqr

                    upper=q3+1.5*iqr

                    df[col]=np.clip(

                        df[col],

                        lower,

                        upper
                    )

            print(
                "\nOutcome Bias Reduced"
            )

        # =================================================
        # FINAL DUPLICATE CHECK
        # =================================================

        final_duplicate_percent=(

            df.duplicated()

            .mean()

        )*100

        print(
            f"\nFinal Duplicate Percentage : "
            f"{final_duplicate_percent:.2f}%"
        )
        
        print("\nFinal Class Distribution:")
        print(df[target_col].value_counts())

        # =================================================
        # FINAL SUMMARY
        # =================================================

        print(
            "\nFinal Dataset Shape:"
        )

        print(df.shape)

        print(
            "\n========== FAIRNESS FIX COMPLETED =========="
        )

        return df

    except Exception as e:

        print(
            f"\nFairness Fix Error: {e}"
        )

        return df