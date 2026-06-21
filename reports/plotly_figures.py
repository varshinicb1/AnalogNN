"""
Plotly-based Figure Generation
==============================

Replaces manual matplotlib plotting with Plotly for interactive,
publication-quality figures with zoom, pan, and export capabilities.
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import torch
from typing import Dict, List, Optional
import os


class PlotlyFigureEngine:
    """
    Interactive figure generation using Plotly.
    """

    def __init__(self, output_dir: str = "./figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_calibration_parity(self, y_ideal: torch.Tensor, y_pre: torch.Tensor,
                                y_post: torch.Tensor, filename: str = "calibration_parity.html") -> str:
        """
        Create parity plot comparing ideal, pre-calibration, and post-calibration outputs.
        """
        y_ideal_np = y_ideal.detach().cpu().numpy().flatten()
        y_pre_np = y_pre.detach().cpu().numpy().flatten()
        y_post_np = y_post.detach().cpu().numpy().flatten()

        fig = go.Figure()

        # Pre-calibration scatter
        fig.add_trace(go.Scatter(
            x=y_ideal_np, y=y_pre_np,
            mode='markers',
            name='Pre-calibration',
            marker=dict(color='red', size=6, opacity=0.6),
            hovertemplate='Ideal: %{x:.3f}<br>Simulated: %{y:.3f}<extra></extra>'
        ))

        # Post-calibration scatter
        fig.add_trace(go.Scatter(
            x=y_ideal_np, y=y_post_np,
            mode='markers',
            name='Post-calibration',
            marker=dict(color='blue', size=6, opacity=0.6),
            hovertemplate='Ideal: %{x:.3f}<br>Calibrated: %{y:.3f}<extra></extra>'
        ))

        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[y_ideal_np.min(), y_ideal_np.max()],
            y=[y_ideal_np.min(), y_ideal_np.max()],
            mode='lines',
            name='Perfect calibration',
            line=dict(color='black', dash='dash', width=2),
            hoverinfo='skip'
        ))

        fig.update_layout(
            title='Calibration Parity Plot',
            xaxis_title='Ideal Output',
            yaxis_title='Simulated/Calibrated Output',
            hovermode='closest',
            template='plotly_white'
        )

        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        return output_path

    def plot_sample_efficiency(self, eff_data: Dict, filename: str = "sample_efficiency.html") -> str:
        """
        Plot calibration RMSE vs training sample size.
        """
        fig = go.Figure()

        for method, data in eff_data.items():
            if isinstance(data, dict) and 'sizes' in data and 'rmse' in data:
                fig.add_trace(go.Scatter(
                    x=data['sizes'],
                    y=data['rmse'],
                    mode='lines+markers',
                    name=method,
                    hovertemplate='Size: %{x}<br>RMSE: %{y:.4f}<extra></extra>'
                ))

        fig.update_layout(
            title='Calibration Sample Efficiency',
            xaxis_title='Training Sample Size',
            yaxis_title='RMSE',
            xaxis_type='log',
            template='plotly_white'
        )

        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        return output_path

    def plot_optimal_resistance_allocation(self, sweep_data: Dict, optimal_rref: float,
                                         filename: str = "optimal_resistance.html") -> str:
        """
        Plot variance vs reference resistance.
        """
        fig = go.Figure()

        if 'r_ref_values' in sweep_data and 'variance' in sweep_data:
            fig.add_trace(go.Scatter(
                x=sweep_data['r_ref_values'],
                y=sweep_data['variance'],
                mode='lines+markers',
                name='Expected Variance',
                hovertemplate='R_ref: %{x:.0f} Ω<br>Variance: %{y:.6f}<extra></extra>'
            ))

            # Mark optimal point
            fig.add_trace(go.Scatter(
                x=[optimal_rref],
                y=[min(sweep_data['variance'])],
                mode='markers',
                name='Optimal R_ref',
                marker=dict(color='red', size=12, symbol='star'),
                hovertemplate='Optimal R_ref: %{x:.0f} Ω<extra></extra>'
            ))

        fig.update_layout(
            title='Optimal Resistance Allocation',
            xaxis_title='Reference Resistance (Ω)',
            yaxis_title='Expected Variance',
            xaxis_type='log',
            template='plotly_white'
        )

        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        return output_path

    def plot_sensitivity_analysis(self, sensitivity_data: Dict,
                                 filename: str = "sensitivity_analysis.html") -> str:
        """
        Plot sensitivity analysis results.
        """
        fig = go.Figure()

        if 'parameters' in sensitivity_data and 'sensitivity' in sensitivity_data:
            fig.add_trace(go.Bar(
                x=sensitivity_data['parameters'],
                y=sensitivity_data['sensitivity'],
                name='Sensitivity',
                hovertemplate='Parameter: %{x}<br>Sensitivity: %{y:.4f}<extra></extra>'
            ))

        fig.update_layout(
            title='Parameter Sensitivity Analysis',
            xaxis_title='Parameter',
            yaxis_title='Sensitivity',
            xaxis_tickangle=-45,
            template='plotly_white'
        )

        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        return output_path

    def plot_residual_histogram(self, residuals: np.ndarray, method_name: str,
                              filename: str = "residual_histogram.html") -> str:
        """
        Plot histogram of calibration residuals.
        """
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=residuals,
            name=f'{method_name} Residuals',
            nbinsx=30,
            hovertemplate='Residual: %{x:.4f}<br>Count: %{y}<extra></extra>'
        ))

        fig.update_layout(
            title=f'Calibration Residuals: {method_name}',
            xaxis_title='Residual',
            yaxis_title='Count',
            template='plotly_white'
        )

        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        return output_path

    def plot_training_history(self, history: Dict, filename: str = "training_history.html") -> str:
        """
        Plot training loss and accuracy over epochs.
        """
        fig = go.Figure()

        if 'train_loss' in history:
            fig.add_trace(go.Scatter(
                x=list(range(len(history['train_loss']))),
                y=history['train_loss'],
                mode='lines',
                name='Training Loss',
                hovertemplate='Epoch: %{x}<br>Loss: %{y:.4f}<extra></extra>'
            ))

        if 'val_loss' in history:
            fig.add_trace(go.Scatter(
                x=list(range(len(history['val_loss']))),
                y=history['val_loss'],
                mode='lines',
                name='Validation Loss',
                hovertemplate='Epoch: %{x}<br>Loss: %{y:.4f}<extra></extra>'
            ))

        fig.update_layout(
            title='Training History',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            template='plotly_white'
        )

        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        return output_path

    def plot_comparison_radar(self, metrics: Dict[str, Dict[str, float]],
                             filename: str = "comparison_radar.html") -> str:
        """
        Create radar chart comparing multiple calibration methods.
        """
        methods = list(metrics.keys())
        metric_names = list(metrics[methods[0]].keys())

        fig = go.Figure()

        for method in methods:
            values = [metrics[method].get(m, 0) for m in metric_names]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metric_names,
                fill='toself',
                name=method
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            title='Calibration Method Comparison',
            template='plotly_white'
        )

        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        return output_path
