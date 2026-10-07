import warnings
warnings.filterwarnings("ignore")

from typing import Any, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

def about_vector(
   vector: Any,
   shape: Optional[bool] = False,
   size: Optional[bool] = False,
   dtype: Optional[bool] = False,
   value: Optional[bool] = False,
   n_dim: Optional[bool] = False
) -> None:
   """Display selected details about a TensorFlow vector."""
   print(f"Vector: \n {vector}")
   print("-"*30)
   if shape:
     print(f"Shape of Vector: {vector.shape}") 
     print("-"*30)
   if size:
     print(f"Size of Vector: {tf.size(vector)}") 
     print("-"*30)
   if dtype:
     print(f"Datatype of Vector: {vector.dtype}") 
     print("-"*30)
   if value:
     print(f"Value of Vector: {vector.numpy()}") 
     print("-"*30)
   if n_dim:
     print(f"Dimention of Vector: {vector.ndim}")


def outlier(df: pd.DataFrame, col: str) -> tuple[float, float]:
    """Return the lower and upper IQR bounds for a DataFrame column."""
    Q1, Q3 = np.percentile(df[col], [25, 75])
    IQR = Q3 - Q1
    minimum = Q1 - 1.5 * IQR
    maximum = Q3 + 1.5 * IQR
    return minimum, maximum


def outlier_plot(
    df: pd.DataFrame,
    feature: str,
    min_val: float,
    max_val: float,
    ax: Any,
) -> None:
    """Plot a feature and highlight values outside the given bounds."""

    is_outlier = (df[feature] < min_val) | (df[feature] > max_val)

    sns.stripplot(
        y=df[feature],
        hue=is_outlier,
        palette={False: "skyblue", True: "orange"},
        ax=ax, 
        color="skyblue", 
        alpha=0.6,
        jitter=0.25,
        legend=False
    )

    ax.axhline(min_val, color="red", linestyle="--", label="Lower bound")
    ax.axhline(max_val, color="red", linestyle="--", label="Upper bound")

    outlier_df = df[is_outlier][feature]
    ax.scatter([], [], color="orange",label="Outliers")
    ax.set_title(f"{feature} with custom bounds")
    ax.legend(loc="upper right")

def outlier_detection(
    df: pd.DataFrame,
    features: Optional[list[str]] = None,
) -> None:
    """Plot outlier detection charts for selected DataFrame features."""
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_features = set(df.columns)
    if features is not None:
        if set(features).issubset(df_features):

            # Grid formula
            N, C = len(features), 3

            rows = (N + C - 1) // C
            _, ax = plt.subplots(rows, C, figsize=(C * 4, 4 * rows))

            ax = np.array(ax).flatten()

            if len(features) == 1:
                ax = [ax]

            for i, feature in enumerate(features):
                min_val, max_val = outlier(df, feature)
                outlier_plot(df, feature, min_val, max_val, ax[i])

            plt.tight_layout()
            plt.show()

        else:
            raise ValueError(f"These features are not in the 'features' parameter: {set(features) - df_features}")
    else:
        # Grid formula
        N, C = len(df_features), 3
        rows = (N + C - 1) // C
        _, ax = plt.subplots(rows, C, figsize=(C * 4, 4 * rows))
        ax = np.array(ax).flatten()
        if len(df_features) == 1:
            ax = [ax]
        for i, feature in enumerate(df_features):
            min_val, max_val = outlier(df, feature)
            outlier_plot(df, feature, min_val, max_val, ax[i])
        plt.tight_layout()
        plt.show

def model_summary(model: Any, history: Any) -> tuple[Any, Any]:
    """Plot training curves and return the model summary and figure."""
    summary = model.summary()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy
    axes[0].plot(acc, label="Training Accuracy")
    axes[0].plot(val_acc, label="Validation Accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Loss
    axes[1].plot(loss, label="Training Loss")
    axes[1].plot(val_loss, label="Validation Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.suptitle(f"Learning Curve of {model.name}")

    return summary, fig


def best_learning_rate(history: Any, epochs:int, model_name: str) -> None:
    """Plot loss across learning rates and highlight the best rate."""
    lrs = 1e-4 * (10**(tf.range(epochs)/20)) 
    losses = history.history['loss'] 

    # Finding the lowest loss and position of the learning rate 
    min_idx = np.argmin(losses) 
    min_lr = lrs[min_idx] 
    min_loss = losses[min_idx] 

    plt.figure(figsize=(8,5)) 
    plt.semilogx(lrs,history.history['loss'], label="Loss") 
    plt.scatter(
        min_lr, 
        min_loss, 
        label=f"Best LR: {np.round(min_lr, 4)} \n Loss: {np.round(min_loss, 3)}", color="red",
    )

    # Annotation
    min_lr_label = np.round(min_lr, 2)
    min_loss_label = np.round(min_loss, 2)
    plt.annotate(
        f"(Lr: {min_lr_label}, Loss: {min_loss_label})",
        xy=(min_lr_label, min_loss_label),
        xytext= (min_lr_label + 1e-1, min_loss_label)
    )
    plt.xlabel("Learning rate") 
    plt.ylabel("Loss") 
    plt.title(f"learning rate v/s Loss for {model_name}")
    plt.legend()
    plt.show()