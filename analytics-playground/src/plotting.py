import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")


def correlation_heatmap(df: pd.DataFrame, figsize: tuple = (10, 8)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig


def distribution_plot(df: pd.DataFrame, column: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_title(f"Distribution of {column}")
    return fig
