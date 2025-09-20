"""
Comprehensive Evaluation Utilities for Fraud Detection
Provides detailed model evaluation, comparison, and business metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation for fraud detection systems
    """
    
    def __init__(self, class_names=None, cost_matrix=None):
        self.class_names = class_names or ['Normal', 'Fraud']
        self.cost_matrix = cost_matrix or {'tp': -100, 'fp': 10, 'fn': 100, 'tn': 0}
        self.evaluation_history = []
        
    def evaluate_binary_classification(self, y_true, y_pred, y_scores=None, 
                                     sample_weight=None, detailed=True):
        """
        Comprehensive binary classification evaluation
        """
        logger.info("Evaluating binary classification performance")
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred, sample_weight=sample_weight),
            'precision': precision_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0),
            'recall': recall_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0),
            'specificity': self._calculate_specificity(y_true, y_pred),
            'sensitivity': recall_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
        }
        
        # Confusion matrix analysis
        cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
        tn, fp, fn, tp = cm.ravel()
        
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0
        })
        
        # AUC metrics if scores provided
        if y_scores is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_scores, sample_weight=sample_weight)
                metrics['pr_auc'] = average_precision_score(y_true, y_scores, sample_weight=sample_weight)
            except ValueError as e:
                logger.warning(f"Could not calculate AUC scores: {e}")
                metrics['roc_auc'] = 0.5
                metrics['pr_auc'] = np.mean(y_true)
        
        # Business metrics
        business_metrics = self._calculate_business_metrics(tn, fp, fn, tp)
        metrics.update(business_metrics)
        
        # Detailed classification report
        if detailed:
            metrics['classification_report'] = classification_report(
                y_true, y_pred, target_names=self.class_names, 
                sample_weight=sample_weight, output_dict=True, zero_division=0
            )
        
        # Store evaluation
        eval_record = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'confusion_matrix': cm.tolist()
        }
        self.evaluation_history.append(eval_record)
        
        return metrics
    
    def _calculate_specificity(self, y_true, y_pred):
        """Calculate specificity (True Negative Rate)"""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def _calculate_business_metrics(self, tn, fp, fn, tp):
        """Calculate business-specific metrics for fraud detection"""
        
        # Cost calculations
        total_cost = (tp * self.cost_matrix['tp'] + 
                     fp * self.cost_matrix['fp'] + 
                     fn * self.cost_matrix['fn'] + 
                     tn * self.cost_matrix['tn'])
        
        # Cost without model (all fraud goes undetected)
        baseline_cost = (tp + fn) * self.cost_matrix['fn']
        
        # Savings
        cost_savings = baseline_cost - total_cost
        savings_rate = cost_savings / abs(baseline_cost) if baseline_cost != 0 else 0
        
        # Detection rates
        fraud_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Efficiency metrics
        precision_fraud = tp / (tp + fp) if (tp + fp) > 0 else 0
        investigation_efficiency = precision_fraud  # Same as precision
        
        return {
            'total_cost': total_cost,
            'baseline_cost': baseline_cost,
            'cost_savings': cost_savings,
            'savings_rate': savings_rate,
            'fraud_detection_rate': fraud_detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'investigation_efficiency': investigation_efficiency,
            'cost_per_investigation': abs(self.cost_matrix['fp']),
            'cost_per_missed_fraud': abs(self.cost_matrix['fn']),
            'revenue_protected': tp * abs(self.cost_matrix['fn'])
        }
    
    def evaluate_threshold_sensitivity(self, y_true, y_scores, thresholds=None):
        """
        Evaluate model performance across different thresholds
        """
        logger.info("Evaluating threshold sensitivity")
        
        if thresholds is None:
            thresholds = np.linspace(0, 1, 101)
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            
            # Skip if all predictions are same class
            if len(np.unique(y_pred)) == 1:
                continue
            
            metrics = self.evaluate_binary_classification(
                y_true, y_pred, y_scores, detailed=False
            )
            metrics['threshold'] = threshold
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        
        # Find optimal thresholds for different metrics
        optimal_thresholds = {}
        key_metrics = ['f1_score', 'precision', 'recall', 'accuracy']
        
        for metric in key_metrics:
            if metric in results_df.columns:
                optimal_idx = results_df[metric].idxmax()
                optimal_thresholds[metric] = {
                    'threshold': results_df.loc[optimal_idx, 'threshold'],
                    'value': results_df.loc[optimal_idx, metric]
                }
        
        return results_df, optimal_thresholds
    
    def cross_validate_model(self, model, X, y, cv_folds=5, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']):
        """
        Perform cross-validation evaluation
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        # Create stratified k-fold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_results = {}
        
        for score_name in scoring:
            try:
                scores = cross_val_score(model, X, y, cv=skf, scoring=score_name, n_jobs=-1)
                cv_results[score_name] = {
                    'scores': scores.tolist(),
                    'mean': float(scores.mean()),
                    'std': float(scores.std()),
                    'min': float(scores.min()),
                    'max': float(scores.max())
                }
                logger.info(f"{score_name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            except Exception as e:
                logger.warning(f"Could not calculate {score_name}: {e}")
                cv_results[score_name] = None
        
        return cv_results
    
    def evaluate_class_imbalance_handling(self, y_true, y_pred, y_scores=None):
        """
        Evaluate how well the model handles class imbalance
        """
        logger.info("Evaluating class imbalance handling")
        
        # Class distribution analysis
        class_distribution = pd.Series(y_true).value_counts(normalize=True)
        imbalance_ratio = class_distribution[0] / class_distribution[1] if len(class_distribution) > 1 else 1
        
        # Performance on minority class (fraud)
        minority_indices = y_true == 1
        majority_indices = y_true == 0
        
        if np.sum(minority_indices) > 0 and np.sum(majority_indices) > 0:
            minority_precision = precision_score(y_true[minority_indices], y_pred[minority_indices], zero_division=0)
            minority_recall = recall_score(y_true[minority_indices], y_pred[minority_indices], zero_division=0)
            majority_precision = precision_score(y_true[majority_indices], y_pred[majority_indices], pos_label=0, zero_division=0)
            majority_recall = recall_score(y_true[majority_indices], y_pred[majority_indices], pos_label=0, zero_division=0)
        else:
            minority_precision = minority_recall = majority_precision = majority_recall = 0
        
        # Balanced accuracy
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        specificity = self._calculate_specificity(y_true, y_pred)
        balanced_accuracy = (sensitivity + specificity) / 2
        
        # Matthews Correlation Coefficient
        try:
            from sklearn.metrics import matthews_corrcoef
            mcc = matthews_corrcoef(y_true, y_pred)
        except:
            mcc = 0
        
        # Geometric mean
        geometric_mean = np.sqrt(sensitivity * specificity)
        
        imbalance_metrics = {
            'imbalance_ratio': float(imbalance_ratio),
            'minority_class_size': int(np.sum(minority_indices)),
            'majority_class_size': int(np.sum(majority_indices)),
            'minority_precision': float(minority_precision),
            'minority_recall': float(minority_recall),
            'majority_precision': float(majority_precision),
            'majority_recall': float(majority_recall),
            'balanced_accuracy': float(balanced_accuracy),
            'mcc': float(mcc),
            'geometric_mean': float(geometric_mean)
        }
        
        return imbalance_metrics
    
    def generate_evaluation_report(self, y_true, y_pred, y_scores=None, model_name="Model", save_path=None):
        """
        Generate comprehensive evaluation report
        """
        logger.info(f"Generating evaluation report for {model_name}")
        
        # Main evaluation
        main_metrics = self.evaluate_binary_classification(y_true, y_pred, y_scores)
        
        # Class imbalance evaluation
        imbalance_metrics = self.evaluate_class_imbalance_handling(y_true, y_pred, y_scores)
        
        # Threshold analysis if scores available
        threshold_analysis = None
        if y_scores is not None:
            threshold_df, optimal_thresholds = self.evaluate_threshold_sensitivity(y_true, y_scores)
            threshold_analysis = {
                'optimal_thresholds': optimal_thresholds,
                'threshold_data': threshold_df.to_dict('records')
            }
        
        # Compile report
        report = {
            'model_name': model_name,
            'evaluation_timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_samples': len(y_true),
                'positive_samples': int(np.sum(y_true)),
                'negative_samples': int(len(y_true) - np.sum(y_true)),
                'positive_ratio': float(np.mean(y_true))
            },
            'performance_metrics': main_metrics,
            'class_imbalance_metrics': imbalance_metrics,
            'threshold_analysis': threshold_analysis
        }
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Evaluation report saved to {save_path}")
        
        # Print summary
        self._print_evaluation_summary(report)
        
        return report
    
    def _print_evaluation_summary(self, report):
        """Print formatted evaluation summary"""
        print("\n" + "="*80)
        print(f"FRAUD DETECTION MODEL EVALUATION REPORT")
        print(f"Model: {report['model_name']}")
        print(f"Date: {report['evaluation_timestamp']}")
        print("="*80)
        
        # Dataset info
        dataset = report['dataset_info']
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   Total Samples: {dataset['total_samples']:,}")
        print(f"   Normal Transactions: {dataset['negative_samples']:,} ({(1-dataset['positive_ratio'])*100:.2f}%)")
        print(f"   Fraud Transactions: {dataset['positive_samples']:,} ({dataset['positive_ratio']*100:.2f}%)")
        
        # Performance metrics
        perf = report['performance_metrics']
        print(f"\n🎯 PERFORMANCE METRICS:")
        print(f"   Accuracy:           {perf['accuracy']:.4f}")
        print(f"   Precision (Fraud):  {perf['precision']:.4f}")
        print(f"   Recall (Fraud):     {perf['recall']:.4f}")
        print(f"   F1-Score:           {perf['f1_score']:.4f}")
        print(f"   Specificity:        {perf['specificity']:.4f}")
        
        if 'roc_auc' in perf:
            print(f"   ROC AUC:            {perf['roc_auc']:.4f}")
        if 'pr_auc' in perf:
            print(f"   PR AUC:             {perf['pr_auc']:.4f}")
        
        # Confusion matrix
        print(f"\n📈 CONFUSION MATRIX:")
        print(f"   True Negatives:     {perf['true_negatives']:,}")
        print(f"   False Positives:    {perf['false_positives']:,}")
        print(f"   False Negatives:    {perf['false_negatives']:,}")
        print(f"   True Positives:     {perf['true_positives']:,}")
        
        # Business metrics
        print(f"\n💰 BUSINESS IMPACT:")
        print(f"   Fraud Detection Rate: {perf['fraud_detection_rate']:.1%}")
        print(f"   False Alarm Rate:     {perf['false_alarm_rate']:.1%}")
        print(f"   Investigation Efficiency: {perf['investigation_efficiency']:.1%}")
        print(f"   Cost Savings:         ${perf['cost_savings']:,.2f}")
        print(f"   Savings Rate:         {perf['savings_rate']:.1%}")
        
        # Class imbalance handling
        imbalance = report['class_imbalance_metrics']
        print(f"\n⚖️ CLASS IMBALANCE HANDLING:")
        print(f"   Imbalance Ratio:      {imbalance['imbalance_ratio']:.1f}:1")
        print(f"   Balanced Accuracy:    {imbalance['balanced_accuracy']:.4f}")
        print(f"   Matthews Correlation: {imbalance['mcc']:.4f}")
        print(f"   Geometric Mean:       {imbalance['geometric_mean']:.4f}")
        
        # Optimal thresholds
        if report['threshold_analysis']:
            print(f"\n🎚️ OPTIMAL THRESHOLDS:")
            for metric, info in report['threshold_analysis']['optimal_thresholds'].items():
                print(f"   {metric.title():15s}: {info['threshold']:.4f} (score: {info['value']:.4f})")
        
        print("="*80)


class ModelComparator:
    """
    Compare multiple fraud detection models
    """
    
    def __init__(self):
        self.comparison_results = []
        
    def compare_models(self, models_results, model_names=None):
        """
        Compare multiple models based on their evaluation results
        
        Args:
            models_results: List of (y_true, y_pred, y_scores) tuples for each model
            model_names: List of model names
        """
        if model_names is None:
            model_names = [f"Model_{i+1}" for i in range(len(models_results))]
        
        logger.info(f"Comparing {len(models_results)} models")
        
        evaluator = ModelEvaluator()
        comparison_data = []
        
        for i, (y_true, y_pred, y_scores) in enumerate(models_results):
            model_name = model_names[i]
            logger.info(f"Evaluating {model_name}")
            
            # Evaluate model
            metrics = evaluator.evaluate_binary_classification(y_true, y_pred, y_scores, detailed=False)
            imbalance_metrics = evaluator.evaluate_class_imbalance_handling(y_true, y_pred, y_scores)
            
            # Combine metrics
            combined_metrics = {**metrics, **imbalance_metrics}
            combined_metrics['model_name'] = model_name
            
            comparison_data.append(combined_metrics)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank models
        key_metrics = ['f1_score', 'roc_auc', 'pr_auc', 'balanced_accuracy', 'cost_savings']
        ranking_results = {}
        
        for metric in key_metrics:
            if metric in comparison_df.columns:
                # Higher is better for these metrics
                ranking_results[metric] = comparison_df.nlargest(len(comparison_df), metric)[['model_name', metric]]
        
        # Overall ranking (simple average of normalized scores)
        normalized_df = comparison_df.copy()
        for metric in key_metrics:
            if metric in normalized_df.columns:
                col_max = normalized_df[metric].max()
                col_min = normalized_df[metric].min()
                if col_max != col_min:
                    normalized_df[f'{metric}_norm'] = (normalized_df[metric] - col_min) / (col_max - col_min)
                else:
                    normalized_df[f'{metric}_norm'] = 1.0
        
        # Calculate overall score
        norm_columns = [f'{metric}_norm' for metric in key_metrics if f'{metric}_norm' in normalized_df.columns]
        if norm_columns:
            normalized_df['overall_score'] = normalized_df[norm_columns].mean(axis=1)
            overall_ranking = normalized_df.nlargest(len(normalized_df), 'overall_score')[['model_name', 'overall_score']]
        else:
            overall_ranking = None
        
        comparison_results = {
            'comparison_data': comparison_df,
            'metric_rankings': ranking_results,
            'overall_ranking': overall_ranking,
            'best_model': overall_ranking.iloc[0]['model_name'] if overall_ranking is not None else model_names[0]
        }
        
        self.comparison_results = comparison_results
        self._print_comparison_summary(comparison_results)
        
        return comparison_results
    
    def _print_comparison_summary(self, results):
        """Print model comparison summary"""
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        # Overall ranking
        if results['overall_ranking'] is not None:
            print("\n🏆 OVERALL RANKING:")
            for i, (_, row) in enumerate(results['overall_ranking'].iterrows()):
                print(f"   {i+1}. {row['model_name']:15s} (Score: {row['overall_score']:.4f})")
        
        # Best performer in each category
        print("\n📊 BEST PERFORMERS BY METRIC:")
        for metric, ranking_df in results['metric_rankings'].items():
            if not ranking_df.empty:
                best_model = ranking_df.iloc[0]
                print(f"   {metric.replace('_', ' ').title():20s}: {best_model['model_name']} ({best_model[metric]:.4f})")
        
        # Detailed comparison table
        print("\n📋 DETAILED COMPARISON:")
        comparison_df = results['comparison_data']
        key_columns = ['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'balanced_accuracy']
        display_columns = [col for col in key_columns if col in comparison_df.columns]
        
        if display_columns:
            print(comparison_df[display_columns].round(4).to_string(index=False))
        
        print(f"\n🎯 RECOMMENDED MODEL: {results['best_model']}")
        print("="*80)
    
    def save_comparison_report(self, save_path):
        """Save detailed comparison report"""
        if not self.comparison_results:
            logger.warning("No comparison results to save")
            return
        
        # Prepare data for JSON serialization
        report_data = {
            'comparison_timestamp': datetime.now().isoformat(),
            'models_compared': len(self.comparison_results['comparison_data']),
            'best_model': self.comparison_results['best_model'],
            'comparison_data': self.comparison_results['comparison_data'].to_dict('records'),
            'metric_rankings': {k: v.to_dict('records') for k, v in self.comparison_results['metric_rankings'].items()},
        }
        
        if self.comparison_results['overall_ranking'] is not None:
            report_data['overall_ranking'] = self.comparison_results['overall_ranking'].to_dict('records')
        
        with open(save_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Comparison report saved to {save_path}")


class BusinessMetricsCalculator:
    """
    Calculate business-specific metrics for fraud detection
    """
    
    def __init__(self, fraud_cost=100, investigation_cost=10, transaction_volume=1000000):
        self.fraud_cost = fraud_cost  # Cost of missing a fraud
        self.investigation_cost = investigation_cost  # Cost of investigating a transaction
        self.transaction_volume = transaction_volume  # Annual transaction volume
    
    def calculate_annual_impact(self, y_true, y_pred, fraud_rate=0.001):
        """
        Calculate annual business impact of fraud detection system
        """
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Scale to annual volume
        test_size = len(y_true)
        scale_factor = self.transaction_volume / test_size
        
        annual_tp = tp * scale_factor
        annual_fp = fp * scale_factor
        annual_fn = fn * scale_factor
        annual_tn = tn * scale_factor
        
        # Cost calculations
        annual_investigation_cost = annual_fp * self.investigation_cost
        annual_missed_fraud_cost = annual_fn * self.fraud_cost
        annual_total_cost = annual_investigation_cost + annual_missed_fraud_cost
        
        # Baseline (no fraud detection system)
        annual_baseline_cost = (annual_tp + annual_fn) * self.fraud_cost
        
        # Savings
        annual_savings = annual_baseline_cost - annual_total_cost
        
        # Revenue protected
        annual_revenue_protected = annual_tp * self.fraud_cost
        
        return {
            'annual_transaction_volume': self.transaction_volume,
            'annual_fraud_transactions': int(annual_tp + annual_fn),
            'annual_fraud_detected': int(annual_tp),
            'annual_fraud_missed': int(annual_fn),
            'annual_false_positives': int(annual_fp),
            'annual_investigation_cost': annual_investigation_cost,
            'annual_missed_fraud_cost': annual_missed_fraud_cost,
            'annual_total_cost': annual_total_cost,
            'annual_baseline_cost': annual_baseline_cost,
            'annual_savings': annual_savings,
            'annual_revenue_protected': annual_revenue_protected,
            'roi_percentage': (annual_savings / annual_investigation_cost * 100) if annual_investigation_cost > 0 else 0
        }
    
    def calculate_detection_efficiency(self, y_true, y_pred, y_scores=None):
        """
        Calculate detection efficiency metrics
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Basic rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        
        # Efficiency metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        alert_rate = (tp + fp) / (tp + fp + tn + fn)
        
        # Lift over random
        baseline_precision = (tp + fn) / (tp + fp + tn + fn)
        lift = precision / baseline_precision if baseline_precision > 0 else 1
        
        return {
            'detection_rate': tpr,
            'false_alarm_rate': fpr,
            'precision': precision,
            'alert_rate': alert_rate,
            'baseline_precision': baseline_precision,
            'lift_over_random': lift,
            'alerts_per_1000_transactions': alert_rate * 1000
        }


# Utility functions
def quick_evaluate(y_true, y_pred, y_scores=None, print_results=True):
    """
    Quick evaluation function with essential metrics
    """
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_binary_classification(y_true, y_pred, y_scores, detailed=False)
    
    if print_results:
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    return metrics


def compare_fraud_detection_models(models_predictions, model_names=None, save_report=None):
    """
    Convenience function for comparing fraud detection models
    """
    comparator = ModelComparator()
    results = comparator.compare_models(models_predictions, model_names)
    
    if save_report:
        comparator.save_comparison_report(save_report)
    
    return results


def calculate_business_roi(y_true, y_pred, fraud_cost=100, investigation_cost=10, annual_volume=1000000):
    """
    Calculate business ROI for fraud detection system
    """
    calculator = BusinessMetricsCalculator(fraud_cost, investigation_cost, annual_volume)
    annual_impact = calculator.calculate_annual_impact(y_true, y_pred)
    efficiency = calculator.calculate_detection_efficiency(y_true, y_pred)
    
    return {**annual_impact, **efficiency}


# Export main classes and functions
__all__ = [
    'ModelEvaluator',
    'ModelComparator', 
    'BusinessMetricsCalculator',
    'quick_evaluate',
    'compare_fraud_detection_models',
    'calculate_business_roi'
]