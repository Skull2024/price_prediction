# src.visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

def plot_summary_graphs(results):
    n_models = len(results)
    n_cols = 2  # Меньше моделей на строку для читаемости
    n_rows = int(np.ceil(n_models / n_cols))

    fig1, axs1 = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    fig2, axs2 = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axs1 = axs1.flatten()
    axs2 = axs2.flatten()

    for i, (name, res) in enumerate(results.items()):
        axs1[i].scatter(res['y_true'], res['y_pred'], alpha=0.5)
        axs1[i].plot([min(res['y_true']), max(res['y_true'])],
                     [min(res['y_true']), max(res['y_true'])], 'r--')
        axs1[i].set_title(f"Actual vs Predicted ({name})")
        axs1[i].set_xlabel("Actual")
        axs1[i].set_ylabel("Predicted")

        residuals = res['y_true'] - res['y_pred']
        sns.histplot(residuals, kde=True, ax=axs2[i])
        axs2[i].set_title(f"Residuals ({name})")
        axs2[i].set_xlabel("Residual")
        axs2[i].set_ylabel("Count")

    for j in range(i+1, len(axs1)):
        axs1[j].axis('off')
        axs2[j].axis('off')

    if "Neural Network" in results:
        fig3, ax3 = plt.subplots(1, 1, figsize=(8, 5))
        ax3.plot(results["Neural Network"]['history']['loss'], label='Train Loss')
        ax3.plot(results["Neural Network"]['history']['val_loss'], label='Val Loss')
        ax3.set_title("Neural Network Loss Curve")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")
        ax3.legend()
        fig3.tight_layout()
        plt.show()

    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()

def plot_metrics_comparison(results):
    metrics_names = ["MSE", "RMSE", "R2", "MedianAE", "MAPE"]
    model_names = list(results.keys())
    metrics_data = {metric: [] for metric in metrics_names}

    for model in model_names:
        for metric in metrics_names:
            metrics_data[metric].append(results[model]["metrics"][metric])

    n_metrics = len(metrics_names)
    fig, axs = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axs = [axs]
    for i, metric in enumerate(metrics_names):
        axs[i].bar(model_names, metrics_data[metric])
        axs[i].set_title(metric)
        axs[i].set_xticks(range(len(model_names)))
        axs[i].set_xticklabels(model_names, rotation=45, ha="right")
        axs[i].set_ylabel(metric)
        # 👉 Удалили подписи над столбцами
    fig.tight_layout()
    plt.show()

def plot_overfitting_comparison(results):
    model_names = [name for name in results.keys() if name != "Neural Network"]
    train_mae = [results[name]["train_metrics"]["MAE"] for name in model_names]
    test_mae = [results[name]["metrics"]["MAE"] for name in model_names]
    cv_mae = [
        results[name].get("cv_metrics", {}).get("MAE", None)
        for name in model_names
    ]


    x = np.arange(len(model_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, train_mae, width, label='Train MAE')
    ax.bar(x, test_mae, width, label='Test MAE')
    ax.bar(x + width, cv_mae, width, label='CV MAE')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_ylabel('MAE')
    ax.set_title('Overfitting Comparison: Train vs Test vs CV MAE')
    ax.legend()
    fig.tight_layout()
    plt.show()

def print_model_ranking(results):
    ranking = sorted(results.items(), key=lambda item: item[1]["metrics"]["MAE"])
    print("\n🏆 Рейтинг моделей по Test MAE:")
    for i, (name, res) in enumerate(ranking, 1):
        print(f"{i}. {name}: {res['metrics']['MAE']:.0f} ₸")

# ✅ Новая функция — тепловая карта метрик моделей
def plot_metrics_heatmap(results, metrics=["MAE", "MAPE", "RMSE"]):
    df = pd.DataFrame([
        {"Model": model, **{metric: results[model]["metrics"][metric] for metric in metrics}}
        # {"Model": model, metric: results[model]["metrics"][metric] for metric in metrics}
        for model in results
    ])
    df.set_index("Model", inplace=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Heatmap of Model Metrics")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.show()

def explain_model_with_shap(model, X_sample, model_name):
    try:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
        plt.title(f"SHAP Summary Plot — {model_name}")
        shap.summary_plot(shap_values, X_sample, show=True)
    except Exception as e:
        print(f"❌ Ошибка при интерпретации SHAP для модели {model_name}: {e}")


def plot_model_ranking(results):
    ranking = sorted(results.items(), key=lambda item: item[1]["metrics"]["MAE"])
    model_names = [name for name, _ in ranking]
    maes = [res["metrics"]["MAE"] for _, res in ranking]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(model_names, maes, color='skyblue')
    ax.set_xlabel("Test MAE")
    ax.set_title("Рейтинг моделей по точности (Test MAE)")
    ax.invert_yaxis()
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 500, bar.get_y() + bar.get_height() / 2,
                f'{width:,.0f} ₸', va='center')
    plt.tight_layout()
    plt.show()
