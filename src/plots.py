"""
Visualization Module for Absenteeism Analysis
Author: Alexis Alduncin (Data Scientist)

Contains all visualization functions for EDA, feature analysis,
and model evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
import logging

from src import config

# Set up logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Set visualization style
plt.style.use(config.PLOT_STYLE)
sns.set_palette(config.COLOR_PALETTE)


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = config.TARGET_COLUMN,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create comprehensive target variable analysis with 4 subplots.

    Args:
        df: Dataframe containing target variable
        target_col: Name of target column
        save_path: Optional path to save figure

    Returns:
        Tuple of (figure, axes)
    """
    logger.info(f"Creating target distribution plot for {target_col}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Target Variable Distribution Analysis', fontsize=16, fontweight='bold')

    # Histogram with mean line
    axes[0, 0].hist(df[target_col], bins=30, color=config.COLOR_PALETTE[0],
                    edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df[target_col].mean(), color='red',
                       linestyle='--', linewidth=2,
                       label=f'Mean: {df[target_col].mean():.2f}h')
    axes[0, 0].axvline(df[target_col].median(), color='green',
                       linestyle='--', linewidth=2,
                       label=f'Median: {df[target_col].median():.2f}h')
    axes[0, 0].set_xlabel('Hours of Absenteeism')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Histogram')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Boxplot
    bp = axes[0, 1].boxplot(df[target_col], patch_artist=True, vert=True)
    bp['boxes'][0].set_facecolor(config.COLOR_PALETTE[1])
    axes[0, 1].set_ylabel('Hours')
    axes[0, 1].set_title('Boxplot - Outlier Detection')
    axes[0, 1].grid(alpha=0.3)

    # Q-Q Plot
    from scipy import stats
    stats.probplot(df[target_col], dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot - Normality Check')
    axes[1, 0].grid(alpha=0.3)

    # Log-transformed distribution
    log_data = np.log1p(df[target_col])
    axes[1, 1].hist(log_data, bins=30, color=config.COLOR_PALETTE[2],
                    edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Log(Hours + 1)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Log-Transformed Distribution')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")

    return fig, axes


def plot_correlation_matrix(
    df: pd.DataFrame,
    method: str = 'pearson',
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create correlation heatmap for numerical features.

    Args:
        df: Dataframe with numerical features
        method: Correlation method ('pearson', 'spearman', 'kendall')
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    logger.info(f"Creating {method} correlation matrix")

    # Select only numerical columns
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr(method=method)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax)

    ax.set_title(f'{method.capitalize()} Correlation Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize feature importance from model.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to display
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    logger.info(f"Plotting top {top_n} feature importances")

    top_features = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=figsize)

    # Create colors gradient
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))

    # Horizontal bar chart
    bars = ax.barh(range(len(top_features)),
                   top_features['importance'].values,
                   color=colors)

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_features['importance'].values)):
        ax.text(val, bar.get_y() + bar.get_height()/2,
                f' {val:.4f}', va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_categorical_analysis(
    df: pd.DataFrame,
    categorical_col: str,
    target_col: str = config.TARGET_COLUMN,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Analyze relationship between categorical feature and target.

    Args:
        df: Dataframe
        categorical_col: Categorical column name
        target_col: Target column name
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    logger.info(f"Creating categorical analysis for {categorical_col}")

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f'Analysis: {categorical_col} vs {target_col}', fontsize=14, fontweight='bold')

    # Count plot
    df[categorical_col].value_counts().plot(kind='bar', ax=axes[0], color=config.COLOR_PALETTE[0])
    axes[0].set_title('Category Counts')
    axes[0].set_xlabel(categorical_col)
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(alpha=0.3)

    # Box plot
    df.boxplot(column=target_col, by=categorical_col, ax=axes[1])
    axes[1].set_title('Distribution by Category')
    axes[1].set_xlabel(categorical_col)
    axes[1].set_ylabel(target_col)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(alpha=0.3)

    # Mean by category
    means = df.groupby(categorical_col)[target_col].mean().sort_values()
    means.plot(kind='barh', ax=axes[2], color=config.COLOR_PALETTE[2])
    axes[2].set_title('Average Absenteeism by Category')
    axes[2].set_xlabel('Mean Hours')
    axes[2].set_ylabel(categorical_col)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_numerical_relationship(
    df: pd.DataFrame,
    num_col: str,
    target_col: str = config.TARGET_COLUMN,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Analyze relationship between numerical feature and target.

    Args:
        df: Dataframe
        num_col: Numerical column name
        target_col: Target column name
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    logger.info(f"Creating numerical analysis for {num_col}")

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f'Analysis: {num_col} vs {target_col}', fontsize=14, fontweight='bold')

    # Histogram
    axes[0].hist(df[num_col], bins=30, color=config.COLOR_PALETTE[0], edgecolor='black', alpha=0.7)
    axes[0].set_title(f'Distribution of {num_col}')
    axes[0].set_xlabel(num_col)
    axes[0].set_ylabel('Frequency')
    axes[0].grid(alpha=0.3)

    # Scatter plot
    axes[1].scatter(df[num_col], df[target_col], alpha=0.5, color=config.COLOR_PALETTE[1])
    axes[1].set_title(f'{num_col} vs {target_col}')
    axes[1].set_xlabel(num_col)
    axes[1].set_ylabel(target_col)
    axes[1].grid(alpha=0.3)

    # Correlation info
    corr = df[[num_col, target_col]].corr().iloc[0, 1]
    axes[1].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                 transform=axes[1].transAxes, fontsize=10,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Binned average
    bins = pd.qcut(df[num_col], q=10, duplicates='drop')
    binned_means = df.groupby(bins)[target_col].mean()
    binned_means.plot(kind='bar', ax=axes[2], color=config.COLOR_PALETTE[2])
    axes[2].set_title('Average Absenteeism by Decile')
    axes[2].set_xlabel(f'{num_col} (binned)')
    axes[2].set_ylabel(f'Mean {target_col}')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize model performance with multiple plots.

    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of model for title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    logger.info(f"Creating performance plots for {model_name}")

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f'{model_name} Performance Analysis', fontsize=14, fontweight='bold')

    # Predicted vs Actual
    axes[0].scatter(y_true, y_pred, alpha=0.5, color=config.COLOR_PALETTE[0])
    axes[0].plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()],
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title('Predicted vs Actual')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Residuals
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, color=config.COLOR_PALETTE[1])
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(alpha=0.3)

    # Residual distribution
    axes[2].hist(residuals, bins=30, color=config.COLOR_PALETTE[2], edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Residuals')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Residual Distribution')
    axes[2].axvline(residuals.mean(), color='red', linestyle='--', lw=2,
                    label=f'Mean: {residuals.mean():.2f}')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")

    return fig


def create_eda_summary_dashboard(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive EDA dashboard with multiple visualizations.

    Args:
        df: Dataframe
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    logger.info("Creating EDA summary dashboard")

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Target distribution
    ax1 = fig.add_subplot(gs[0, :])
    df[config.TARGET_COLUMN].hist(bins=50, ax=ax1, color=config.COLOR_PALETTE[0], edgecolor='black')
    ax1.set_title('Target Variable Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Absenteeism Hours')
    ax1.set_ylabel('Frequency')

    # Day of week analysis
    ax2 = fig.add_subplot(gs[1, 0])
    df.groupby('Day of the week')[config.TARGET_COLUMN].mean().plot(kind='bar', ax=ax2, color=config.COLOR_PALETTE[1])
    ax2.set_title('Avg Absence by Day of Week')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Mean Hours')

    # Season analysis
    ax3 = fig.add_subplot(gs[1, 1])
    df.groupby('Seasons')[config.TARGET_COLUMN].mean().plot(kind='bar', ax=ax3, color=config.COLOR_PALETTE[2])
    ax3.set_title('Avg Absence by Season')
    ax3.set_xlabel('Season')
    ax3.set_ylabel('Mean Hours')

    # Age distribution
    ax4 = fig.add_subplot(gs[1, 2])
    df['Age'].hist(bins=20, ax=ax4, color=config.COLOR_PALETTE[3], edgecolor='black')
    ax4.set_title('Age Distribution')
    ax4.set_xlabel('Age')
    ax4.set_ylabel('Count')

    # Distance vs Absence
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.scatter(df['Distance from Residence to Work'], df[config.TARGET_COLUMN],
                alpha=0.5, color=config.COLOR_PALETTE[4])
    ax5.set_title('Distance vs Absence')
    ax5.set_xlabel('Distance (km)')
    ax5.set_ylabel('Absence Hours')

    # BMI vs Absence
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.scatter(df['Body mass index'], df[config.TARGET_COLUMN],
                alpha=0.5, color=config.COLOR_PALETTE[0])
    ax6.set_title('BMI vs Absence')
    ax6.set_xlabel('BMI')
    ax6.set_ylabel('Absence Hours')

    # Workload vs Absence
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.scatter(df['Work load Average/day'], df[config.TARGET_COLUMN],
                alpha=0.5, color=config.COLOR_PALETTE[1])
    ax7.set_title('Workload vs Absence')
    ax7.set_xlabel('Workload')
    ax7.set_ylabel('Absence Hours')

    fig.suptitle('Exploratory Data Analysis Dashboard', fontsize=18, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved dashboard to {save_path}")

    return fig


if __name__ == "__main__":
    print("Visualization Module - Absenteeism Analysis")
    print("Available functions:")
    print("  - plot_target_distribution()")
    print("  - plot_correlation_matrix()")
    print("  - plot_feature_importance()")
    print("  - plot_categorical_analysis()")
    print("  - plot_numerical_relationship()")
    print("  - plot_model_performance()")
    print("  - create_eda_summary_dashboard()")
