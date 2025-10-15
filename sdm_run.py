
"""
sdm_run.py — single-file SDM pipeline based on the workshop notebook

- Loads PA shapefiles + current rasters
- Preprocesses train/test and full target stacks
- Trains RF/ET/XGB/LGBM with 5-fold CV, prints metrics
- Predicts rasters (current + SSP2) per model
- Builds ensemble (mean), writes merged rasters
- Writes difference (SSP2 - current), plots PNG
- Produces summary tables, dot-whisker plots, and PDPs
"""

# ----------------------- imports -----------------------
import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import geopandas as gpd
import rasterio

from pyimpute import impute
from pyimpute import load_training_vector
from pyimpute import load_targets

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import model_selection as mod_sel
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.inspection import partial_dependence
from sklearn.metrics import confusion_matrix as evaluate_conf

# ----------------------- config -----------------------
# Edit these to your machine:
BASE_DIR = "/Users/carissamayo/Desktop/UW_work/EEID_AI/Coding"

PA_FILES = [
    os.path.join(BASE_DIR, "presence-absence/oligoryzomys_longicaudatus_(bennett_1832)_pca_ratio_1_0.shp"),
    os.path.join(BASE_DIR, "presence-absence/oligoryzomys_longicaudatus_(bennett_1832)_pca_ratio_2_0.shp"),
    os.path.join(BASE_DIR, "presence-absence/oligoryzomys_longicaudatus_(bennett_1832)_random_ratio_1_0.shp"),
    os.path.join(BASE_DIR, "presence-absence/oligoryzomys_longicaudatus_(bennett_1832)_random_ratio_2_0.shp"),
]

CURR_RASTER_DIR = os.path.join(BASE_DIR, "Clipped_current_rasters_O")
FUTR_RASTER_DIR = os.path.join(BASE_DIR, "Clipped_future_rasters_O")
OUTPUT_BASE     = BASE_DIR  # where OutputO* folders are written

RESPONSE_FIELD = "species"  # field in PA shapefiles used for stratification

FEATURE_NAMES = [
    "Annual Mean Temperature",
    "Mean Temperature of Warmest Quarter",
    "Mean Temperature of Coldest Quarter",
    "Annual Precipitation",
    "Precipitation of Wettest Month",
    "Precipitation of Driest Month",
    "Precipitation Seasonality",
    "Precipitation of Wettest Quarter",
    "Precipitation of Driest Quarter",
    "Precipitation of Warmest Quarter",
    "Precipitation of Coldest Quarter",
    "Mean Diurnal Range",
    "Isothermality",
    "Temperature Seasonality",
    "Max Temperature of Warmest Month",
    "Min Temperature of Coldest Month",
    "Temperature Annual Range",
    "Mean Temperature of Wettest Quarter",
    "Mean Temperature of Driest Quarter",
]

# ----------------------- helpers -----------------------
def load_data(pa_path, raster_dir):
    pa = gpd.GeoDataFrame.from_file(pa_path)
    raster_features = sorted(glob.glob(os.path.join(raster_dir, "*.tif")))
    if len(raster_features) == 0:
        raise FileNotFoundError(f"No .tif rasters found in {raster_dir}")
    return pa, raster_features

def preprocess_data(pa, raster_features, response_field=RESPONSE_FIELD, test_size=0.25):
    pa_train, pa_test = train_test_split(
        pa, test_size=test_size, stratify=pa[response_field], random_state=42
    )
    # training features/labels
    train_xs, train_y = load_training_vector(pa_train, raster_features, response_field=response_field)
    # full-coverage targets for prediction
    target_xs, raster_info = load_targets(raster_features)

    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    train_xs = imputer.fit_transform(train_xs)
    target_xs[np.isnan(target_xs)] = 0

    # test set (kept consistent with your notebook approach)
    test_xs, test_y = load_training_vector(pa_test, raster_features, response_field=response_field)
    test_xs = imputer.fit_transform(test_xs)
    return train_xs, train_y, test_xs, test_y, target_xs, raster_info

def train_and_predict(train_xs, train_y, test_xs, test_y,
                      target_xs, raster_info, target_ssp2, ssp2_info,
                      output_base_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)

    CLASS_MAP = {
        "rf":  RandomForestClassifier(),
        "et":  ExtraTreesClassifier(),
        "xgb": XGBClassifier(),
        "lgbm": LGBMClassifier(verbose=-1),
    }

    trained_models = {}
    model_metrics  = {}

    for name, model in CLASS_MAP.items():
        print(f"\n=== {name.upper()} ===")
        kf = mod_sel.KFold(n_splits=5, shuffle=True, random_state=42)

        # CV metrics
        for metric in ["accuracy", "roc_auc", "precision", "recall"]:
            scores = mod_sel.cross_val_score(model, train_xs, train_y, cv=kf, scoring=metric)
            mean_pct = scores.mean() * 100
            sd2_pct  = scores.std()  * 200  # std*2 *100
            print(f"CV {metric:9}: {mean_pct:5.2f} (+/- {sd2_pct:5.2f})")

        # Fit + test confusion matrix
        model.fit(train_xs, train_y)
        y_pred = model.predict(test_xs)
        conf = evaluate_conf(test_y, y_pred)
        print(f"Confusion matrix (test):\n{conf}")

        # Keep recall/precision for tables
        rec_scores = mod_sel.cross_val_score(model, train_xs, train_y, cv=kf, scoring="recall")
        pre_scores = mod_sel.cross_val_score(model, train_xs, train_y, cv=kf, scoring="precision")
        model_metrics[name] = {
            "recall_mean":    rec_scores.mean() * 100,
            "recall_sd":      rec_scores.std() * 200,
            "precision_mean": pre_scores.mean() * 100,
            "precision_sd":   pre_scores.std() * 200,
        }

        # Raster predictions (current)
        dir_curr = os.path.join(output_base_dir, f"{name}-images_noRFE_current")
        os.makedirs(dir_curr, exist_ok=True)
        impute(target_xs, model, raster_info, outdir=dir_curr, class_prob=True, certainty=True)

        # Raster predictions (SSP2)
        dir_ssp2 = os.path.join(output_base_dir, f"{name}-images_noRFE_ssp2")
        os.makedirs(dir_ssp2, exist_ok=True)
        impute(target_ssp2, model, ssp2_info, outdir=dir_ssp2, class_prob=True, certainty=True)

        trained_models[name] = model

    return model_metrics, trained_models

def save_table_as_image(df, filename, title=None, fontsize=12, cell_height=0.4, cell_width=1.5):
    fig, ax = plt.subplots(figsize=(cell_width * df.shape[1] + 2, cell_height * df.shape[0] + 2))
    ax.axis('off')
    table = ax.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.2, 1.2)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

def prepare_long_df(mean_df, sd_df):
    df    = mean_df.reset_index()
    df_sd = sd_df.reset_index()
    df    = df.rename(columns={df.columns[0]: 'PA_Method'})
    df_sd = df_sd.rename(columns={df_sd.columns[0]: 'PA_Method'})
    mean_long = df.melt(id_vars='PA_Method', var_name='Model', value_name='Mean')
    sd_long   = df_sd.melt(id_vars='PA_Method', var_name='Model', value_name='SD')
    long_df   = mean_long.copy()
    long_df['SD'] = sd_long['SD'].values
    return long_df

def plot_dot_whisker(long_df, title, filename):
    pa_methods = long_df['PA_Method'].unique()
    models     = long_df['Model'].unique()
    n_groups   = len(pa_methods)
    n_models   = len(models)
    group_width = 0.8
    model_width = group_width / n_models

    fig, ax = plt.subplots(figsize=(12, 6))
    indices = np.arange(n_groups)

    for i, model in enumerate(models):
        x_pos = indices - group_width/2 + i*model_width + model_width/2
        means, sds = [], []
        for pa in pa_methods:
            subset = long_df[(long_df['PA_Method'] == pa) & (long_df['Model'] == model)]
            if not subset.empty:
                means.append(subset['Mean'].values[0])
                sds.append(subset['SD'].values[0])
            else:
                means.append(np.nan); sds.append(np.nan)

        means, sds = np.array(means), np.array(sds)
        line, = ax.plot(x_pos, means, 'o', label=model)
        color = line.get_color()
        ax.vlines(x_pos, means - sds, means + sds, color=color, alpha=0.7)
        cap = model_width * 0.2
        ax.hlines(means - sds, x_pos - cap, x_pos + cap, color=color, alpha=0.7)
        ax.hlines(means + sds, x_pos - cap, x_pos + cap, color=color, alpha=0.7)

    ax.set_xticks(indices)
    ax.set_xticklabels(pa_methods, rotation=30, ha='right')
    ax.set_title(title); ax.set_ylabel('Score (Mean ± SD)')
    ax.legend(title='Model'); ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.show()

def analyze_model(trained_models, feature_names, train_xs):
    top_features_per_model = {}
    importances_per_model  = {}
    for name, model in trained_models.items():
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            sorted_df = df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
            top5_df = sorted_df.head(5)
            top_features_per_model[name] = top5_df['Feature'].tolist()
            importances_per_model[name]  = top5_df
        else:
            top_features_per_model[name] = ["N/A"]
            importances_per_model[name]  = pd.DataFrame()
    return top_features_per_model, importances_per_model

def plot_pdp_grid(run_group_name, run_numbers, all_trained_models, all_train_xs, feature_names, out_path=None):
    model_keys = ["rf", "et", "xgb", "lgbm"]
    run_to_pa = {
        1: "1:1 PCA", 2: "1:2 PCA", 3: "1:1 Random", 4: "1:2 Random",
        5: "1:1 PCA", 6: "1:2 PCA", 7: "1:1 Random", 8: "1:2 Random",
    }

    fig, axs = plt.subplots(2, 2, figsize=(22, 22))
    axs = axs.flatten()

    n_rows, n_cols = len(run_numbers), 3
    for model_idx, model_key in enumerate(model_keys):
        ax_main = axs[model_idx]
        ax_main.axis('off')
        pos = ax_main.get_position()
        width  = 1.0 / n_cols
        height = 1.0 / n_rows

        for row_idx, run in enumerate(run_numbers):
            pa_label = run_to_pa.get(run, f"Run {run}")
            model = all_trained_models[run].get(model_key)
            if model is None or not hasattr(model, "feature_importances_"):
                continue

            X_run = pd.DataFrame(all_train_xs[run], columns=feature_names)
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[::-1][:3]
            top_features = X_run.columns[top_indices]

            # row label
            plt.figtext(
                pos.x0 - 0.04,
                pos.y0 + (1 - (row_idx + 0.5) * height) * pos.height,
                pa_label, va='center', ha='right', fontsize=13, rotation=90, fontweight='bold'
            )

            for col_idx, feature in enumerate(top_features):
                left = col_idx * width
                bottom = 1 - (row_idx + 1) * height
                ax = plt.axes([
                    pos.x0 + left * pos.width,
                    pos.y0 + bottom * pos.height,
                    width * pos.width * 0.95,
                    height * pos.height * 0.85
                ])

                pd_result = partial_dependence(model, X_run, [feature], kind="average")
                grid_values = pd_result.grid_values[0]
                pdp_mean    = pd_result.average[0]
                ax.plot(grid_values, pdp_mean)
                ax.set_title(f"{feature}", fontsize=9)
                if col_idx == 0: ax.set_ylabel("Partial dependence", fontsize=9)
                else: ax.set_yticklabels([])
                ax.tick_params(axis='both', which='major', labelsize=8)
                tick_idx = np.linspace(0, len(grid_values) - 1, num=5, dtype=int)
                ax.set_xticks(grid_values[tick_idx])
                ax.set_xticklabels([f"{grid_values[i]:.2f}" for i in tick_idx], rotation=45, fontsize=8)

    plt.suptitle(f"PDPs for Top 3 Features\n{run_group_name}", fontsize=22)
    plt.subplots_adjust(left=0.18, right=0.97, top=0.93, bottom=0.08, hspace=1.3, wspace=0.2)
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.show()
    return fig

# ----------------------- main -----------------------
def main():
    all_trained_models = {}
    all_train_xs = {}

    recall_vals, recall_sd_vals = [], []
    prec_vals,   prec_sd_vals   = [], []

    model_order = ['rf', 'et', 'xgb', 'lgbm']
    file_labels = ["1:1 PA, PCA_based", "1:2 PA, PCA_based",
                   "1:1 PA, random sampling", "1:2 PA, random sampling"]

    # iterate runs
    for idx, pa_path in enumerate(PA_FILES, start=1):
        print(f"\n########## RUN {idx}: {os.path.basename(pa_path)} ##########")
        pa, raster_features = load_data(pa_path, CURR_RASTER_DIR)
        train_xs, train_y, test_xs, test_y, target_xs, raster_info = preprocess_data(pa, raster_features)

        # SSP2 targets
        ssp2_paths = sorted(glob.glob(os.path.join(FUTR_RASTER_DIR, "*.tif")))
        if not ssp2_paths:
            raise FileNotFoundError(f"No SSP2 rasters in {FUTR_RASTER_DIR}")
        target_ssp2, ssp2_info = load_targets(ssp2_paths)
        target_ssp2[np.isnan(target_ssp2)] = 0

        # per-run outdir
        out_dir = os.path.join(OUTPUT_BASE, f"OutputO{idx}")
        os.makedirs(out_dir, exist_ok=True)

        # train + predict
        metrics, models = train_and_predict(
            train_xs, train_y, test_xs, test_y,
            target_xs, raster_info, target_ssp2, ssp2_info,
            output_base_dir=out_dir
        )

        # store
        all_trained_models[idx] = models
        all_train_xs[idx] = train_xs

        # collect metrics in RF/ET/XGB/LGBM order
        for m in model_order:
            recall_vals.append(metrics[m]['recall_mean'])
            recall_sd_vals.append(metrics[m]['recall_sd'])
            prec_vals.append(metrics[m]['precision_mean'])
            prec_sd_vals.append(metrics[m]['precision_sd'])

        # ---- ensemble and diff for this run ----
        def _r(folder):
            return rasterio.open(os.path.join(out_dir, folder, "probability_1.tif")).read(1)

        try:
            curr_stack = np.stack([
                _r("rf-images_noRFE_current"),
                _r("et-images_noRFE_current"),
                _r("xgb-images_noRFE_current"),
                _r("lgbm-images_noRFE_current"),
            ], axis=0)
            ssp2_stack = np.stack([
                _r("rf-images_noRFE_ssp2"),
                _r("et-images_noRFE_ssp2"),
                _r("xgb-images_noRFE_ssp2"),
                _r("lgbm-images_noRFE_ssp2"),
            ], axis=0)
        except Exception as e:
            print(f"Ensemble read warning: {e}")
            continue

        distr_curr = np.nanmean(curr_stack, axis=0)
        distr_ssp2 = np.nanmean(ssp2_stack, axis=0)

        # copy meta from rf current
        rf_curr_path = os.path.join(out_dir, "rf-images_noRFE_current", "probability_1.tif")
        meta = rasterio.open(rf_curr_path).meta.copy()

        with rasterio.open(os.path.join(out_dir, "merged_SDM.tif"), 'w', **meta) as dst:
            dst.write(distr_curr.astype(rasterio.float32), 1)
        with rasterio.open(os.path.join(out_dir, "merged_ssp2_SDM.tif"), 'w', **meta) as dst:
            dst.write(distr_ssp2.astype(rasterio.float32), 1)

        diff = distr_ssp2 - distr_curr
        with rasterio.open(os.path.join(out_dir, "diff_SDM.tif"), 'w', **meta) as dst:
            dst.write(diff.astype(rasterio.float32), 1)

        plt.figure()
        plt.imshow(diff, cmap="RdBu", interpolation="nearest")
        plt.colorbar()
        plt.title("future − current species distribution probabilities", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "diff_SDM_plot.png"), dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()

    # ---- summary tables & plots (O runs) ----
    if recall_vals:
        cols = ['RF','ET','XGB','LGBM']
        recall_df = pd.DataFrame(np.array(recall_vals).reshape(-1, 4), columns=cols, index=file_labels)
        recall_sd = pd.DataFrame(np.array(recall_sd_vals).reshape(-1, 4), columns=cols, index=file_labels)
        prec_df   = pd.DataFrame(np.array(prec_vals).reshape(-1, 4), columns=cols, index=file_labels)
        prec_sd   = pd.DataFrame(np.array(prec_sd_vals).reshape(-1, 4), columns=cols, index=file_labels)

        recall_summary    = recall_df.round(1).astype(str) + " (± " + recall_sd.round(1).astype(str) + ")"
        precision_summary = prec_df.round(1).astype(str)   + " (± " + prec_sd.round(1).astype(str) + ")"

        save_table_as_image(recall_summary,    os.path.join(OUTPUT_BASE, "recall_summary.png"),    title="Recall (± SD)")
        save_table_as_image(precision_summary, os.path.join(OUTPUT_BASE, "precision_summary.png"), title="Precision (± SD)")

        # dot-whisker
        recall_long = prepare_long_df(recall_df, recall_sd)
        prec_long   = prepare_long_df(prec_df, prec_sd)
        plot_dot_whisker(recall_long, "Recall",    os.path.join(OUTPUT_BASE, "recall_O_plot.png"))
        plot_dot_whisker(prec_long,   "Precision", os.path.join(OUTPUT_BASE, "precision_O_plot.png"))
    else:
        print("No metrics collected — check data paths and raster/PA availability.")

    # ---- Feature importances + PDPs for runs 1 & 3 ----
    if all_trained_models:
        try:
            out_path = os.path.join(OUTPUT_BASE, "O_pdp_plot.png")
            plot_pdp_grid("Oligoryzomys longicaudatus", [1, 3],
                          all_trained_models, all_train_xs, FEATURE_NAMES,
                          out_path=out_path)
        except Exception as e:
            print(f"PDP plotting skipped: {e}")

if __name__ == "__main__":
    main()
