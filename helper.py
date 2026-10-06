import warnings
warnings.filterwarnings("ignore")

from typing import Optional
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

def about_vector(
   vector, 
   shape: Optional[bool] = False,
   size: Optional[bool] = False,
   dtype: Optional[bool] = False,
   value: Optional[bool] = False,
   n_dim: Optional[bool] = False
) -> None:
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




def outlier(df: pd.DataFrame, col: str):
    Q1, Q3 = np.percentile(df[col], [25, 75])
    IQR = Q3 - Q1
    minimum = Q1 - 1.5 * IQR
    maximum = Q3 + 1.5 * IQR
    return minimum, maximum

def outlier_plot(df, feature, min_val, max_val, ax):

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
    features: Optional[list] = None
):
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

def model_summary(model, history):
    summary = model.summary()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(acc, label="Training Accuracy")
    ax.plot(val_acc, label="Validation Accuracy")
    ax.plot(loss, label="Training Loss")
    ax.plot(val_loss, label="Validation Loss")
    ax.set_title(f"Learning Curve of {model.name}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss / Accuracy")
    ax.legend()

    return summary, fig
