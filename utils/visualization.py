"""
Advanced Visualization Utilities for Fraud Detection
Provides comprehensive plotting and interactive visualization capabilities
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (confusion_matrix, precision_recall_curve, roc_curve, 
                           auc, classification_report)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
import logging

# Setup
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class FraudVisualization:
    """
    Comprehensive visualization toolkit for fraud detection analysis
    """
    
    def __init__(self, style='seaborn-v0_8', figsize=(15, 10), color_palette='husl'):
        plt.style.use(style)
        sns.set_palette(color_palette)
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_dataset_overview(self, df, save_path=None):
        """
        Plot comprehensive dataset overview
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Credit Card Fraud Dataset Overview', fontsize=16, y=1.02)
        
        # 1. Class distribution (pie chart)
        if 'Class' in df.columns:
            class_counts = df['Class'].value_counts()
            colors_pie = ['lightblue', 'lightcoral']
            wedges, texts, autotexts = axes[0, 0].pie(
                class_counts.values, labels=['Normal', 'Fraud'], 
                autopct='%1.1f%%', colors=colors_pie, startangle=90
            )
            axes[0, 0].set_title('Transaction Class Distribution')
            
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        # 2. Amount distribution by class
        if 'Amount' in df.columns and 'Class' in df.columns:
            normal_amounts = df[df['Class'] == 0]['Amount']
            fraud_amounts = df[df['Class'] == 1]['Amount']
            
            axes[0, 1].hist(normal_amounts, bins=50, alpha=0.7, label='Normal', 
                           density=True, color=self.colors[0])
            axes[0, 1].hist(fraud_amounts, bins=50, alpha=0.7, label='Fraud',
                           density=True, color=self.colors[1])
            axes[0, 1].set_title('Transaction Amount Distribution')
            axes[0, 1].set_xlabel('Amount')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].legend()
            axes[0, 1].set_xlim(0, np.percentile(df['Amount'], 95))
        
        # 3. Time distribution (if available)
        if 'Time' in df.columns:
            time_hours = (df['Time'] / 3600) % 24
            axes[0, 2].hist(time_hours, bins=24, alpha=0.7, color=self.colors[2])
            axes[0, 2].set_title('Transactions by Hour of Day')
            axes[0, 2].set_xlabel('Hour')
            axes[0, 2].set_ylabel('Count')
            axes[0, 2].set_xticks(range(0, 25, 4))
        
        # 4. Feature correlation heatmap (sample)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        sample_cols = list(numeric_cols[:8])  # First 8 numeric columns
        if 'Class' in df.columns and 'Class' not in sample_cols:
            sample_cols.append('Class')
        
        if len(sample_cols) > 1:
            corr_matrix = df[sample_cols].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, ax=axes[1, 0], fmt='.2f', square=True)
            axes[1, 0].set_title('Feature Correlation Matrix (Sample)')
        
        # 5. Feature distribution comparison
        if len(numeric_cols) > 2:
            feature = numeric_cols[1]  # Second numeric feature
            if 'Class' in df.columns:
                normal_vals = df[df['Class'] == 0][feature]
                fraud_vals = df[df['Class'] == 1][feature]
                
                axes[1, 1].hist(normal_vals, bins=30, alpha=0.7, label='Normal', 
                               density=True, color=self.colors[0])
                axes[1, 1].hist(fraud_vals, bins=30, alpha=0.7, label='Fraud',
                               density=True, color=self.colors[1])
                axes[1, 1].set_title(f'{feature} Distribution by Class')
                axes[1, 1].set_xlabel(f'{feature} Value')
                axes[1, 1].set_ylabel('Density')
                axes[1, 1].legend()
        
        # 6. Dataset statistics summary
        axes[1, 2].axis('off')
        stats_text = f"""
        Dataset Statistics:
        
        Total Samples: {len(df):,}
        Features: {len(df.columns)-1 if 'Class' in df.columns else len(df.columns)}
        Memory Usage: {df.memory_usage(deep=True).sum()/1024**2:.1f} MB
        
        Missing Values: {df.isnull().sum().sum()}
        Duplicate Rows: {df.duplicated().sum()}
        """
        
        if 'Class' in df.columns:
            fraud_count = df['Class'].sum()
            fraud_ratio = df['Class'].mean()
            stats_text += f"""
        Normal Transactions: {len(df) - fraud_count:,}
        Fraud Transactions: {fraud_count:,}
        Fraud Ratio: {fraud_ratio:.4f} ({fraud_ratio*100:.2f}%)
        """
        
        axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dataset overview saved to {save_path}")
        
        plt.show()
    
    def plot_model_training_history(self, history, save_path=None):
        """
        Plot model training history with multiple metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Model Training History', fontsize=16)
        
        epochs = range(1, len(history['total_loss']) + 1)
        
        # Total loss
        axes[0, 0].plot(epochs, history['total_loss'], 'b-', label='Total Loss', linewidth=2)
        if 'val_total_loss' in history:
            axes[0, 0].plot(epochs, history['val_total_loss'], 'r--', label='Val Total Loss', linewidth=2)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reconstruction loss
        if 'reconstruction_loss' in history:
            axes[0, 1].plot(epochs, history['reconstruction_loss'], 'g-', 
                           label='Reconstruction Loss', linewidth=2)
            if 'val_reconstruction_loss' in history:
                axes[0, 1].plot(epochs, history['val_reconstruction_loss'], 'orange', 
                               label='Val Reconstruction Loss', linewidth=2)
            axes[0, 1].set_title('Reconstruction Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # LMP loss
        if 'lmp_loss' in history:
            axes[1, 0].plot(epochs, history['lmp_loss'], 'purple', 
                           label='LMP Loss', linewidth=2)
            if 'val_lmp_loss' in history:
                axes[1, 0].plot(epochs, history['val_lmp_loss'], 'brown', 
                               label='Val LMP Loss', linewidth=2)
            axes[1, 0].set_title('Loss Prediction Module Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate (if available)
        if 'lr' in history:
            axes[1, 1].plot(epochs, history['lr'], 'orange', linewidth=2)
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Loss components comparison
            axes[1, 1].plot(epochs, history['total_loss'], label='Total', linewidth=2)
            if 'reconstruction_loss' in history:
                axes[1, 1].plot(epochs, history['reconstruction_loss'], 
                               label='Reconstruction', linewidth=2)
            if 'lmp_loss' in history:
                axes[1, 1].plot(epochs, history['lmp_loss'], label='LMP', linewidth=2)
            axes[1, 1].set_title('Loss Components Comparison')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history saved to {save_path}")
        
        plt.show()
    
    def plot_fraud_detection_results(self, X_test, y_test, predictions, errors, 
                                   predicted_losses, threshold, save_path=None):
        """
        Comprehensive fraud detection results visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Fraud Detection System Results', fontsize=16)
        
        # 1. Reconstruction error distribution
        normal_errors = errors[y_test == 0]
        fraud_errors = errors[y_test == 1]
        
        axes[0, 0].hist(normal_errors, bins=50, alpha=0.7, label='Normal', 
                       density=True, color=self.colors[0])
        axes[0, 0].hist(fraud_errors, bins=50, alpha=0.7, label='Fraud',
                       density=True, color=self.colors[1])
        axes[0, 0].axvline(threshold, color='red', linestyle='--', linewidth=2, 
                          label=f'Threshold ({threshold:.4f})')
        axes[0, 0].set_title('Reconstruction Error Distribution')
        axes[0, 0].set_xlabel('Reconstruction Error')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].set_xlim(0, np.percentile(errors, 95))
        
        # 2. Predicted vs Actual Loss scatter plot
        predicted_losses_flat = predicted_losses.flatten()
        axes[0, 1].scatter(errors[y_test == 0], predicted_losses_flat[y_test == 0], 
                          alpha=0.5, s=1, color=self.colors[0], label='Normal')
        axes[0, 1].scatter(errors[y_test == 1], predicted_losses_flat[y_test == 1], 
                          alpha=0.8, s=3, color=self.colors[1], label='Fraud')
        
        # Perfect prediction line
        max_error = max(np.max(errors), np.max(predicted_losses_flat))
        axes[0, 1].plot([0, max_error], [0, max_error], 'r--', alpha=0.5, 
                       label='Perfect Prediction')
        
        axes[0, 1].set_title('Predicted vs Actual Reconstruction Error')
        axes[0, 1].set_xlabel('Actual Reconstruction Error')
        axes[0, 1].set_ylabel('Predicted Loss')
        axes[0, 1].legend()
        
        # 3. Confusion Matrix
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2],
                   xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
        axes[0, 2].set_title('Confusion Matrix')
        axes[0, 2].set_xlabel('Predicted')
        axes[0, 2].set_ylabel('Actual')
        
        # Add accuracy text
        accuracy = np.sum(predictions == y_test) / len(y_test)
        axes[0, 2].text(0.5, -0.1, f'Accuracy: {accuracy:.3f}', 
                       transform=axes[0, 2].transAxes, ha='center')
        
        # 4. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, errors)
        roc_auc = auc(fpr, tpr)
        
        axes[1, 0].plot(fpr, tpr, color=self.colors[0], linewidth=2,
                       label=f'ROC Curve (AUC = {roc_auc:.3f})')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        axes[1, 0].set_title('ROC Curve')
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, errors)
        pr_auc = auc(recall, precision)
        
        axes[1, 1].plot(recall, precision, color=self.colors[2], linewidth=2,
                       label=f'PR Curve (AUC = {pr_auc:.3f})')
        axes[1, 1].set_title('Precision-Recall Curve')
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Error distribution by sample index
        normal_indices = np.where(y_test == 0)[0]
        fraud_indices = np.where(y_test == 1)[0]
        
        axes[1, 2].scatter(normal_indices, normal_errors, alpha=0.5, s=1,
                          color=self.colors[0], label='Normal')
        axes[1, 2].scatter(fraud_indices, fraud_errors, alpha=0.8, s=3,
                          color=self.colors[1], label='Fraud')
        axes[1, 2].axhline(threshold, color='red', linestyle='--', linewidth=2,
                          label='Threshold')
        axes[1, 2].set_title('Reconstruction Error by Sample')
        axes[1, 2].set_xlabel('Sample Index')
        axes[1, 2].set_ylabel('Reconstruction Error')
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Detection results saved to {save_path}")
        
        plt.show()
        
        # Print performance summary
        self._print_performance_summary(y_test, predictions, errors, roc_auc, pr_auc)
    
    def _print_performance_summary(self, y_test, predictions, errors, roc_auc, pr_auc):
        """Print performance summary"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC AUC:   {roc_auc:.4f}")
        print(f"PR AUC:    {pr_auc:.4f}")
        
        # Confusion matrix values
        cm = confusion_matrix(y_test, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Positives:  {tp}")
        
        print("="*60)
    
    def plot_active_learning_progress(self, al_history, baseline_fraud_rate=None, save_path=None):
        """
        Plot active learning progress over iterations
        """
        if not al_history:
            logger.warning("No active learning history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Active Learning Progress', fontsize=16)
        
        iterations = [item['iteration'] for item in al_history]
        fraud_ratios = [item['fraud_ratio'] for item in al_history]
        uncertainties = [item['avg_uncertainty'] for item in al_history]
        cumulative_labeled = [item.get('total_labeled', item['iteration'] * item['n_selected']) 
                             for item in al_history]
        
        # 1. Fraud detection rate over iterations
        axes[0, 0].plot(iterations, fraud_ratios, 'o-', linewidth=2, markersize=8, 
                       color=self.colors[1], label='Selected Samples')
        
        if baseline_fraud_rate:
            axes[0, 0].axhline(baseline_fraud_rate, color='gray', linestyle='--', 
                              label=f'Random Baseline ({baseline_fraud_rate:.3f})')
        
        axes[0, 0].set_title('Fraud Detection Rate in Selected Samples')
        axes[0, 0].set_xlabel('Active Learning Iteration')
        axes[0, 0].set_ylabel('Fraud Ratio in Selection')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, max(fraud_ratios) * 1.1)
        
        # 2. Average uncertainty scores
        axes[0, 1].plot(iterations, uncertainties, 's-', linewidth=2, markersize=8, 
                       color=self.colors[2])
        axes[0, 1].set_title('Average Uncertainty Score')
        axes[0, 1].set_xlabel('Active Learning Iteration')
        axes[0, 1].set_ylabel('Average Uncertainty')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Cumulative labeling progress
        axes[1, 0].plot(iterations, cumulative_labeled, '^-', linewidth=2, markersize=8, 
                       color=self.colors[3])
        axes[1, 0].set_title('Cumulative Labeled Samples')
        axes[1, 0].set_xlabel('Active Learning Iteration')
        axes[1, 0].set_ylabel('Total Samples Labeled')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Efficiency comparison
        if baseline_fraud_rate:
            efficiency = [fr / baseline_fraud_rate for fr in fraud_ratios]
            axes[1, 1].bar(iterations, efficiency, color=self.colors[4], alpha=0.7)
            axes[1, 1].axhline(1.0, color='red', linestyle='--', alpha=0.7, 
                              label='Random Selection')
            axes[1, 1].set_title('Selection Efficiency vs Random')
            axes[1, 1].set_xlabel('Active Learning Iteration')
            axes[1, 1].set_ylabel('Efficiency Ratio')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Active learning progress saved to {save_path}")
        
        plt.show()
        
        # Print summary
        if baseline_fraud_rate:
            avg_efficiency = np.mean([fr / baseline_fraud_rate for fr in fraud_ratios])
            print(f"\nActive Learning Summary:")
            print(f"  Average fraud detection rate: {np.mean(fraud_ratios):.3f}")
            print(f"  Average efficiency vs random: {avg_efficiency:.2f}x")
            print(f"  Total samples labeled: {cumulative_labeled[-1]}")
    
    def plot_feature_importance(self, feature_importance, feature_names=None, 
                               top_n=20, save_path=None):
        """
        Plot feature importance analysis
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
        
        # Sort by importance
        sorted_indices = np.argsort(feature_importance)[-top_n:]
        sorted_importance = feature_importance[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(range(len(sorted_importance)), sorted_importance, 
                      color=self.colors[0], alpha=0.8)
        
        ax.set_yticks(range(len(sorted_importance)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, sorted_importance)):
            ax.text(importance + max(sorted_importance) * 0.01, i, 
                   f'{importance:.4f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_threshold_analysis(self, y_true, y_scores, save_path=None):
        """
        Plot threshold analysis for optimal threshold selection
        """
        thresholds = np.linspace(0, np.max(y_scores), 100)
        
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            
            # Skip if all predictions are the same class
            if len(np.unique(y_pred)) == 1:
                accuracies.append(0)
                precisions.append(0)
                recalls.append(0)
                f1_scores.append(0)
                continue
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracies.append(accuracy_score(y_true, y_pred))
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Threshold Analysis for Optimal Performance', fontsize=16)
        
        # Plot each metric
        metrics = [
            (accuracies, 'Accuracy', self.colors[0]),
            (precisions, 'Precision', self.colors[1]),
            (recalls, 'Recall', self.colors[2]),
            (f1_scores, 'F1-Score', self.colors[3])
        ]
        
        for i, (metric_values, metric_name, color) in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            ax.plot(thresholds, metric_values, linewidth=2, color=color)
            ax.set_title(f'{metric_name} vs Threshold')
            ax.set_xlabel('Threshold')
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3)
            
            # Mark optimal point
            if metric_values:
                optimal_idx = np.argmax(metric_values)
                optimal_threshold = thresholds[optimal_idx]
                optimal_value = metric_values[optimal_idx]
                
                ax.scatter(optimal_threshold, optimal_value, color='red', s=100, 
                          zorder=5, label=f'Optimal: {optimal_threshold:.4f}')
                ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Threshold analysis saved to {save_path}")
        
        plt.show()
        
        # Return optimal thresholds
        optimal_thresholds = {}
        for metric_values, metric_name, _ in metrics:
            if metric_values:
                optimal_idx = np.argmax(metric_values)
                optimal_thresholds[metric_name.lower()] = thresholds[optimal_idx]
        
        return optimal_thresholds
    
    def plot_latent_space_visualization(self, encoded_features, labels, method='tsne', save_path=None):
        """
        Visualize latent space using dimensionality reduction
        """
        if encoded_features.shape[1] < 2:
            logger.warning("Encoded features have less than 2 dimensions, skipping visualization")
            return
        
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            title = 't-SNE Visualization of Latent Space'
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            title = 'PCA Visualization of Latent Space'
        else:
            logger.error(f"Unknown dimensionality reduction method: {method}")
            return
        
        # Reduce dimensionality
        reduced_features = reducer.fit_transform(encoded_features)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot normal and fraud samples
        normal_mask = labels == 0
        fraud_mask = labels == 1
        
        ax.scatter(reduced_features[normal_mask, 0], reduced_features[normal_mask, 1], 
                  c=self.colors[0], alpha=0.6, s=20, label='Normal')
        ax.scatter(reduced_features[fraud_mask, 0], reduced_features[fraud_mask, 1], 
                  c=self.colors[1], alpha=0.8, s=30, label='Fraud')
        
        ax.set_title(title)
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Latent space visualization saved to {save_path}")
        
        plt.show()


class InteractiveFraudVisualization:
    """
    Interactive Plotly-based visualizations for fraud detection
    """
    
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def create_interactive_dashboard(self, X_test, y_test, errors, predictions, 
                                   predicted_losses, threshold):
        """
        Create comprehensive interactive dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Error Distribution', 'Error vs Index', 'Confusion Matrix',
                          'ROC Curve', 'Precision-Recall Curve', 'Predicted vs Actual Loss'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. Error distribution
        normal_errors = errors[y_test == 0]
        fraud_errors = errors[y_test == 1]
        
        fig.add_trace(go.Histogram(x=normal_errors, name='Normal', opacity=0.7,
                                 nbinsx=50, histnorm='probability density',
                                 marker_color=self.colors[0]), row=1, col=1)
        fig.add_trace(go.Histogram(x=fraud_errors, name='Fraud', opacity=0.7,
                                 nbinsx=50, histnorm='probability density',
                                 marker_color=self.colors[1]), row=1, col=1)
        
        # Add threshold line
        fig.add_vline(x=threshold, line_dash="dash", line_color="red", row=1, col=1)
        
        # 2. Error vs Index scatter
        indices = np.arange(len(errors))
        colors_scatter = ['Normal' if y == 0 else 'Fraud' for y in y_test]
        
        fig.add_trace(go.Scatter(x=indices, y=errors, mode='markers',
                               marker=dict(color=y_test, colorscale=[[0, self.colors[0]], [1, self.colors[1]]],
                                         size=3, opacity=0.6),
                               name='Samples',
                               text=[f'Class: {c}' for c in colors_scatter],
                               hovertemplate='Index: %{x}<br>Error: %{y:.4f}<br>%{text}'),
                     row=1, col=2)
        
        # Add threshold line
        fig.add_hline(y=threshold, line_dash="dash", line_color="red", row=1, col=2)
        
        # 3. Confusion Matrix
        cm = confusion_matrix(y_test, predictions)
        fig.add_trace(go.Heatmap(z=cm, x=['Normal', 'Fraud'], y=['Normal', 'Fraud'],
                               colorscale='Blues', showscale=False,
                               text=cm, texttemplate="%{text}", textfont={"size": 20}),
                     row=1, col=3)
        
        # 4. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, errors)
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={roc_auc:.3f})',
                               line=dict(width=3, color=self.colors[2])), row=2, col=1)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                               line=dict(dash='dash', color='gray', width=2),
                               name='Random', showlegend=False), row=2, col=1)
        
        # 5. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, errors)
        pr_auc = auc(recall, precision)
        
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR (AUC={pr_auc:.3f})',
                               line=dict(width=3, color=self.colors[3])), row=2, col=2)
        
        # 6. Predicted vs Actual Loss
        predicted_losses_flat = predicted_losses.flatten()
        
        fig.add_trace(go.Scatter(x=errors[y_test == 0], y=predicted_losses_flat[y_test == 0],
                               mode='markers', name='Normal',
                               marker=dict(size=3, color=self.colors[0], opacity=0.6)),
                     row=2, col=3)
        fig.add_trace(go.Scatter(x=errors[y_test == 1], y=predicted_losses_flat[y_test == 1],
                               mode='markers', name='Fraud',
                               marker=dict(size=5, color=self.colors[1], opacity=0.8)),
                     row=2, col=3)
        
        # Perfect prediction line
        max_error = max(np.max(errors), np.max(predicted_losses_flat))
        fig.add_trace(go.Scatter(x=[0, max_error], y=[0, max_error], mode='lines',
                               line=dict(dash='dash', color='red', width=2),
                               name='Perfect', showlegend=False), row=2, col=3)
        
        # Update layout
        fig.update_layout(height=800, showlegend=True,
                         title_text="Interactive Fraud Detection Analysis Dashboard")
        
        # Update axes labels
        fig.update_xaxes(title_text="Reconstruction Error", row=1, col=1)
        fig.update_xaxes(title_text="Sample Index", row=1, col=2)
        fig.update_xaxes(title_text="Predicted", row=1, col=3)
        fig.update_xaxes(title_text="False Positive Rate", row=2, col=1)
        fig.update_xaxes(title_text="Recall", row=2, col=2)
        fig.update_xaxes(title_text="Actual Error", row=2, col=3)
        
        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_yaxes(title_text="Reconstruction Error", row=1, col=2)
        fig.update_yaxes(title_text="Actual", row=1, col=3)
        fig.update_yaxes(title_text="True Positive Rate", row=2, col=1)
        fig.update_yaxes(title_text="Precision", row=2, col=2)
        fig.update_yaxes(title_text="Predicted Loss", row=2, col=3)
        
        return fig
    
    def create_3d_latent_space(self, encoded_features, labels):
        """
        Create 3D visualization of latent space
        """
        if encoded_features.shape[1] < 3:
            # Use PCA to get 3D representation
            pca = PCA(n_components=3)
            features_3d = pca.fit_transform(encoded_features)
        else:
            features_3d = encoded_features[:, :3]
        
        normal_mask = labels == 0
        fraud_mask = labels == 1
        
        fig = go.Figure()
        
        # Normal samples
        fig.add_trace(go.Scatter3d(
            x=features_3d[normal_mask, 0],
            y=features_3d[normal_mask, 1],
            z=features_3d[normal_mask, 2],
            mode='markers',
            marker=dict(size=3, color=self.colors[0], opacity=0.6),
            name='Normal'
        ))
        
        # Fraud samples
        fig.add_trace(go.Scatter3d(
            x=features_3d[fraud_mask, 0],
            y=features_3d[fraud_mask, 1],
            z=features_3d[fraud_mask, 2],
            mode='markers',
            marker=dict(size=5, color=self.colors[1], opacity=0.8),
            name='Fraud'
        ))
        
        fig.update_layout(
            title='3D Latent Space Visualization',
            scene=dict(
                xaxis_title='Latent Dimension 1',
                yaxis_title='Latent Dimension 2',
                zaxis_title='Latent Dimension 3'
            )
        )
        
        return fig


# Utility functions
def save_all_plots(visualization, results_dict, save_dir='plots'):
    """
    Save all visualization plots to directory
    """
    import os
    from datetime import datetime
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Dataset overview
    if 'df' in results_dict:
        visualization.plot_dataset_overview(
            results_dict['df'], 
            save_path=f"{save_dir}/dataset_overview_{timestamp}.png"
        )
    
    # Training history
    if 'history' in results_dict:
        visualization.plot_model_training_history(
            results_dict['history'],
            save_path=f"{save_dir}/training_history_{timestamp}.png"
        )
    
    # Detection results
    if all(key in results_dict for key in ['X_test', 'y_test', 'predictions', 'errors']):
        visualization.plot_fraud_detection_results(
            results_dict['X_test'], results_dict['y_test'], 
            results_dict['predictions'], results_dict['errors'],
            results_dict.get('predicted_losses'), results_dict.get('threshold'),
            save_path=f"{save_dir}/detection_results_{timestamp}.png"
        )
    
    # Active learning progress
    if 'al_history' in results_dict:
        visualization.plot_active_learning_progress(
            results_dict['al_history'],
            baseline_fraud_rate=results_dict.get('baseline_fraud_rate'),
            save_path=f"{save_dir}/active_learning_{timestamp}.png"
        )
    
    logger.info(f"All plots saved to {save_dir}/")


def create_report_plots(results, output_dir='report_plots'):
    """
    Create a comprehensive set of plots for reporting
    """
    viz = FraudVisualization()
    save_all_plots(viz, results, output_dir)
    
    # Create interactive dashboard
    interactive_viz = InteractiveFraudVisualization()
    dashboard = interactive_viz.create_interactive_dashboard(
        results['X_test'], results['y_test'], results['predictions'],
        results['errors'], results.get('predicted_losses'), results.get('threshold')
    )
    
    dashboard.write_html(f"{output_dir}/interactive_dashboard.html")
    logger.info(f"Interactive dashboard saved to {output_dir}/interactive_dashboard.html")


# Export main classes and functions
__all__ = [
    'FraudVisualization',
    'InteractiveFraudVisualization',
    'save_all_plots',
    'create_report_plots'
]