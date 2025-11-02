"""
Visualization utilities for Texas Crash Analysis Dashboard
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import warnings
from typing import Optional

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
warnings.filterwarnings('ignore', category=FutureWarning, module='plotly')


def create_severity_pie_chart(df: pd.DataFrame, severity_col: str = 'Severity') -> go.Figure:
    """
    Create a pie chart showing severity distribution

    Args:
        df: DataFrame with crash data
        severity_col: Name of severity column

    Returns:
        Plotly figure
    """
    severity_counts = df[severity_col].value_counts().sort_index()

    fig = go.Figure(data=[go.Pie(
        labels=[f'Severity {int(s)}' for s in severity_counts.index],
        values=severity_counts.values,
        hole=0.3,
        marker=dict(colors=['#2ecc71', '#f39c12', '#e74c3c', '#c0392b'])
    )])

    fig.update_layout(
        title='Crash Severity Distribution',
        height=400,
        showlegend=True
    )

    return fig


def create_temporal_line_chart(df: pd.DataFrame, date_col: str = 'Start_Time',
                                freq: str = 'ME', title: str = 'Crashes Over Time') -> go.Figure:
    """
    Create a line chart showing temporal patterns

    Args:
        df: DataFrame with temporal data
        date_col: Name of date column
        freq: Frequency for resampling ('D', 'W', 'M', 'Y')
        title: Chart title

    Returns:
        Plotly figure
    """
    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
    df_temp = df_temp.dropna(subset=[date_col])

    # Resample by frequency
    temporal_counts = df_temp.set_index(date_col).resample(freq).size()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=temporal_counts.index,
        y=temporal_counts.values,
        mode='lines+markers',
        name='Crashes',
        line=dict(color='#3498db', width=2),
        marker=dict(size=4)
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Number of Crashes',
        height=400,
        hovermode='x unified'
    )

    return fig


def create_county_bar_chart(df: pd.DataFrame, county_col: str = 'County',
                             top_n: int = 15, title: str = 'Top Counties by Crash Count') -> go.Figure:
    """
    Create a bar chart showing top counties

    Args:
        df: DataFrame with crash data
        county_col: Name of county column
        top_n: Number of top counties to show
        title: Chart title

    Returns:
        Plotly figure
    """
    county_counts = df[county_col].value_counts().head(top_n)

    fig = go.Figure(data=[go.Bar(
        x=county_counts.index,
        y=county_counts.values,
        marker=dict(
            color=county_counts.values,
            colorscale='Blues',
            showscale=False
        )
    )])

    fig.update_layout(
        title=title,
        xaxis_title='County',
        yaxis_title='Number of Crashes',
        height=400,
        xaxis_tickangle=-45
    )

    return fig


def create_correlation_heatmap(df: pd.DataFrame, features: Optional[list] = None,
                                title: str = 'Feature Correlation Heatmap') -> go.Figure:
    """
    Create a correlation heatmap for numerical features

    Args:
        df: DataFrame with features
        features: List of features to include (None for all numeric)
        title: Chart title

    Returns:
        Plotly figure
    """
    if features is None:
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        numeric_df = df[features]

    # Compute correlation matrix
    corr_matrix = numeric_df.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title='Correlation')
    ))

    fig.update_layout(
        title=title,
        height=600,
        width=700,
        xaxis_tickangle=-45
    )

    return fig


def create_feature_histogram(df: pd.DataFrame, feature: str,
                              bins: int = 50, title: Optional[str] = None) -> go.Figure:
    """
    Create a histogram for a feature

    Args:
        df: DataFrame with feature
        feature: Name of feature
        bins: Number of bins
        title: Chart title

    Returns:
        Plotly figure
    """
    if title is None:
        title = f'Distribution of {feature}'

    fig = go.Figure(data=[go.Histogram(
        x=df[feature].dropna(),
        nbinsx=bins,
        marker=dict(color='#3498db', line=dict(color='white', width=1))
    )])

    fig.update_layout(
        title=title,
        xaxis_title=feature,
        yaxis_title='Count',
        height=400,
        showlegend=False
    )

    return fig


def create_risk_distribution_chart(df: pd.DataFrame, risk_col: str = 'risk_category') -> go.Figure:
    """
    Create a bar chart showing risk category distribution

    Args:
        df: DataFrame with risk categories
        risk_col: Name of risk column

    Returns:
        Plotly figure
    """
    risk_counts = df[risk_col].value_counts()

    # Define colors for risk levels
    color_map = {
        'LOW': '#2ecc71',
        'MEDIUM': '#f39c12',
        'HIGH': '#e67e22',
        'VERY_HIGH': '#e74c3c'
    }

    colors = [color_map.get(risk, '#95a5a6') for risk in risk_counts.index]

    fig = go.Figure(data=[go.Bar(
        x=risk_counts.index,
        y=risk_counts.values,
        marker=dict(color=colors),
        text=risk_counts.values,
        textposition='auto'
    )])

    fig.update_layout(
        title='Risk Category Distribution',
        xaxis_title='Risk Level',
        yaxis_title='Number of Segments',
        height=400,
        showlegend=False
    )

    return fig


def create_time_series_with_trend(df: pd.DataFrame, date_col: str, value_col: str,
                                    title: str = 'Time Series with Trend') -> go.Figure:
    """
    Create a time series plot with trend line

    Args:
        df: DataFrame with temporal data
        date_col: Name of date column
        value_col: Name of value column
        title: Chart title

    Returns:
        Plotly figure
    """
    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
    df_temp = df_temp.dropna(subset=[date_col, value_col])
    df_temp = df_temp.sort_values(date_col)

    # Create figure
    fig = go.Figure()

    # Add actual values
    fig.add_trace(go.Scatter(
        x=df_temp[date_col],
        y=df_temp[value_col],
        mode='lines',
        name='Actual',
        line=dict(color='#3498db', width=2)
    ))

    # Add trend line (simple moving average)
    window = min(30, len(df_temp) // 10)  # Adaptive window
    if window > 1:
        df_temp['trend'] = df_temp[value_col].rolling(window=window, center=True).mean()
        fig.add_trace(go.Scatter(
            x=df_temp[date_col],
            y=df_temp['trend'],
            mode='lines',
            name='Trend',
            line=dict(color='#e74c3c', width=2, dash='dash')
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=value_col,
        height=400,
        hovermode='x unified'
    )

    return fig


def create_box_plot(df: pd.DataFrame, category_col: str, value_col: str,
                    title: Optional[str] = None) -> go.Figure:
    """
    Create a box plot comparing categories

    Args:
        df: DataFrame with data
        category_col: Name of category column
        value_col: Name of value column
        title: Chart title

    Returns:
        Plotly figure
    """
    if title is None:
        title = f'{value_col} by {category_col}'

    fig = px.box(
        df,
        x=category_col,
        y=value_col,
        color=category_col,
        title=title
    )

    fig.update_layout(
        height=400,
        showlegend=False
    )

    return fig
