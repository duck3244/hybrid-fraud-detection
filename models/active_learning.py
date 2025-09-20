"""
Active Learning Strategies for Fraud Detection
Implements various uncertainty-based sampling strategies
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import pairwise_distances
from abc import ABC, abstractmethod
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActiveLearningStrategy(ABC):
    """
    Abstract base class for active learning strategies
    """
    
    def __init__(self, model, strategy_name="base"):
        self.model = model
        self.strategy_name = strategy_name
        self.selection_history = []
        
    @abstractmethod
    def select_samples(self, X_unlabeled, n_samples, **kwargs):
        """Select most informative samples for labeling"""
        pass
    
    def update_history(self, selected_indices, scores):
        """Update selection history"""
        self.selection_history.append({
            'iteration': len(self.selection_history) + 1,
            'selected_indices': selected_indices,
            'scores': scores,
            'n_selected': len(selected_indices)
        })


class UncertaintyBasedStrategy(ActiveLearningStrategy):
    """
    Uncertainty-based active learning using reconstruction error and LPM predictions
    """
    
    def __init__(self, model, threshold=0.5, combination_method='multiply'):
        super().__init__(model, "uncertainty_based")
        self.threshold = threshold
        self.combination_method = combination_method
        
    def select_samples(self, X_unlabeled, n_samples, **kwargs):
        """
        Select samples based on uncertainty scores
        """
        logger.info(f"Selecting {n_samples} samples using uncertainty-based strategy")
        
        # Get model predictions
        reconstructed, predicted_losses = self.model(X_unlabeled, training=False)
        
        # Calculate reconstruction errors
        reconstruction_errors = tf.reduce_mean(tf.square(X_unlabeled - reconstructed), axis=1)
        predicted_losses = tf.squeeze(predicted_losses)
        
        # Combine scores based on method
        if self.combination_method == 'multiply':
            uncertainty_scores = reconstruction_errors * predicted_losses
        elif self.combination_method == 'add':
            uncertainty_scores = reconstruction_errors + predicted_losses
        elif self.combination_method == 'weighted':
            alpha = kwargs.get('alpha', 0.7)
            uncertainty_scores = alpha * reconstruction_errors + (1 - alpha) * predicted_losses
        else:
            uncertainty_scores = reconstruction_errors
        
        # Select top uncertain samples
        top_indices = tf.nn.top_k(uncertainty_scores, k=min(n_samples, len(X_unlabeled))).indices
        
        selected_indices = top_indices.numpy()
        scores = uncertainty_scores.numpy()
        
        self.update_history(selected_indices, scores)
        
        return selected_indices, scores


class DiversityBasedStrategy(ActiveLearningStrategy):
    """
    Diversity-based active learning to ensure representative sampling
    """
    
    def __init__(self, model, diversity_weight=0.5):
        super().__init__(model, "diversity_based")
        self.diversity_weight = diversity_weight
        
    def select_samples(self, X_unlabeled, n_samples, **kwargs):
        """
        Select samples balancing uncertainty and diversity
        """
        logger.info(f"Selecting {n_samples} samples using diversity-based strategy")
        
        # Get uncertainty scores
        uncertainty_strategy = UncertaintyBasedStrategy(self.model)
        _, uncertainty_scores = uncertainty_strategy.select_samples(
            X_unlabeled, len(X_unlabeled), **kwargs
        )
        
        # Get encoded representations for diversity calculation
        encoded_features = self.model.encode(X_unlabeled, training=False)
        
        # Select samples using diversity-aware algorithm
        selected_indices = self._select_diverse_samples(
            encoded_features.numpy(), uncertainty_scores, n_samples
        )
        
        self.update_history(selected_indices, uncertainty_scores)
        
        return selected_indices, uncertainty_scores
    
    def _select_diverse_samples(self, features, uncertainty_scores, n_samples):
        """
        Select diverse samples using farthest-first traversal
        """
        selected_indices = []
        remaining_indices = list(range(len(features)))
        
        # Start with most uncertain sample
        first_idx = np.argmax(uncertainty_scores)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Select remaining samples
        for _ in range(min(n_samples - 1, len(remaining_indices))):
            max_score = -1
            next_idx = None
            
            for idx in remaining_indices:
                # Calculate minimum distance to selected samples
                if selected_indices:
                    distances = pairwise_distances(
                        features[idx:idx+1], features[selected_indices]
                    )[0]
                    min_distance = np.min(distances)
                else:
                    min_distance = 0
                
                # Combine uncertainty and diversity
                combined_score = (self.diversity_weight * min_distance + 
                                (1 - self.diversity_weight) * uncertainty_scores[idx])
                
                if combined_score > max_score:
                    max_score = combined_score
                    next_idx = idx
            
            if next_idx is not None:
                selected_indices.append(next_idx)
                remaining_indices.remove(next_idx)
        
        return np.array(selected_indices)


class QBCStrategy(ActiveLearningStrategy):
    """
    Query by Committee (QBC) strategy using ensemble disagreement
    """
    
    def __init__(self, model, n_committee=5):
        super().__init__(model, "qbc")
        self.n_committee = n_committee
        
    def select_samples(self, X_unlabeled, n_samples, **kwargs):
        """
        Select samples based on committee disagreement
        """
        logger.info(f"Selecting {n_samples} samples using QBC strategy")
        
        # Generate predictions from multiple committee members
        committee_predictions = []
        
        for i in range(self.n_committee):
            # Add noise to create committee diversity
            noise_std = kwargs.get('noise_std', 0.01)
            X_noisy = X_unlabeled + tf.random.normal(tf.shape(X_unlabeled), 0, noise_std)
            
            # Get predictions
            reconstructed, predicted_losses = self.model(X_noisy, training=True)  # Use training=True for dropout
            reconstruction_errors = tf.reduce_mean(tf.square(X_noisy - reconstructed), axis=1)
            
            committee_predictions.append(reconstruction_errors.numpy())
        
        # Calculate disagreement (variance across committee)
        committee_predictions = np.array(committee_predictions)
        disagreement_scores = np.var(committee_predictions, axis=0)
        
        # Select top disagreement samples
        top_indices = np.argsort(disagreement_scores)[-n_samples:]
        
        self.update_history(top_indices, disagreement_scores)
        
        return top_indices, disagreement_scores


class AdaptiveStrategy(ActiveLearningStrategy):
    """
    Adaptive strategy that switches between different methods based on performance
    """
    
    def __init__(self, model, strategies=None):
        super().__init__(model, "adaptive")
        
        if strategies is None:
            self.strategies = {
                'uncertainty': UncertaintyBasedStrategy(model),
                'diversity': DiversityBasedStrategy(model),
                'qbc': QBCStrategy(model)
            }
        else:
            self.strategies = strategies
        
        self.strategy_performance = {name: [] for name in self.strategies.keys()}
        self.current_strategy = 'uncertainty'
        
    def select_samples(self, X_unlabeled, n_samples, **kwargs):
        """
        Select samples using adaptive strategy selection
        """
        logger.info(f"Selecting {n_samples} samples using adaptive strategy ({self.current_strategy})")
        
        # Use current best strategy
        strategy = self.strategies[self.current_strategy]
        selected_indices, scores = strategy.select_samples(X_unlabeled, n_samples, **kwargs)
        
        # Update history
        self.update_history(selected_indices, scores)
        
        return selected_indices, scores
    
    def update_strategy_performance(self, strategy_name, performance_score):
        """Update performance tracking for strategies"""
        self.strategy_performance[strategy_name].append(performance_score)
        
        # Switch to best performing strategy
        avg_performances = {}
        for name, scores in self.strategy_performance.items():
            if scores:
                avg_performances[name] = np.mean(scores[-5:])  # Last 5 iterations
        
        if avg_performances:
            self.current_strategy = max(avg_performances, key=avg_performances.get)
            logger.info(f"Switched to strategy: {self.current_strategy}")


class CostSensitiveStrategy(ActiveLearningStrategy):
    """
    Cost-sensitive active learning considering annotation costs
    """
    
    def __init__(self, model, cost_function=None):
        super().__init__(model, "cost_sensitive")
        self.cost_function = cost_function or self._default_cost_function
        
    def _default_cost_function(self, X_samples):
        """Default cost function - uniform cost"""
        return np.ones(len(X_samples))
        
    def select_samples(self, X_unlabeled, n_samples, budget=None, **kwargs):
        """
        Select samples considering annotation costs
        """
        logger.info(f"Selecting samples using cost-sensitive strategy (budget: {budget})")
        
        # Get uncertainty scores
        uncertainty_strategy = UncertaintyBasedStrategy(self.model)
        _, uncertainty_scores = uncertainty_strategy.select_samples(
            X_unlabeled, len(X_unlabeled), **kwargs
        )
        
        # Calculate annotation costs
        costs = self.cost_function(X_unlabeled)
        
        # Calculate cost-effectiveness ratio
        cost_effectiveness = uncertainty_scores / (costs + 1e-8)
        
        # Select samples within budget
        if budget is not None:
            selected_indices = self._select_within_budget(
                cost_effectiveness, costs, budget, n_samples
            )
        else:
            # Select top cost-effective samples
            selected_indices = np.argsort(cost_effectiveness)[-n_samples:]
        
        self.update_history(selected_indices, cost_effectiveness)
        
        return selected_indices, cost_effectiveness
    
    def _select_within_budget(self, effectiveness, costs, budget, max_samples):
        """Select samples within budget constraint"""
        # Sort by effectiveness
        sorted_indices = np.argsort(effectiveness)[::-1]
        
        selected_indices = []
        current_cost = 0
        
        for idx in sorted_indices:
            if len(selected_indices) >= max_samples:
                break
            if current_cost + costs[idx] <= budget:
                selected_indices.append(idx)
                current_cost += costs[idx]
        
        return np.array(selected_indices)


class ActiveLearningManager:
    """
    Manager class for running active learning experiments
    """
    
    def __init__(self, model, strategy_type='uncertainty', **strategy_kwargs):
        self.model = model
        self.strategy = self._create_strategy(strategy_type, **strategy_kwargs)
        self.experiment_history = []
        
    def _create_strategy(self, strategy_type, **kwargs):
        """Factory method for creating strategies"""
        strategies = {
            'uncertainty': UncertaintyBasedStrategy,
            'diversity': DiversityBasedStrategy,
            'qbc': QBCStrategy,
            'adaptive': AdaptiveStrategy,
            'cost_sensitive': CostSensitiveStrategy
        }

        if strategy_type not in strategies:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

        return strategies[strategy_type](self.model, **kwargs)

    def run_iteration(self, X_unlabeled, y_unlabeled, n_samples, **kwargs):
        """
        Run single active learning iteration
        """
        # Select samples
        selected_indices, scores = self.strategy.select_samples(
            X_unlabeled, n_samples, **kwargs
        )

        # Simulate labeling (in practice, this would be manual)
        selected_labels = y_unlabeled[selected_indices]

        # Calculate iteration metrics
        iteration_metrics = {
            'iteration': len(self.experiment_history) + 1,
            'selected_indices': selected_indices,
            'n_selected': len(selected_indices),
            'fraud_ratio': np.mean(selected_labels),
            'avg_uncertainty': np.mean(scores[selected_indices]),
            'total_labeled': sum([exp['n_selected'] for exp in self.experiment_history]) + len(selected_indices)
        }

        self.experiment_history.append(iteration_metrics)

        logger.info(f"Iteration {iteration_metrics['iteration']} completed:")
        logger.info(f"  Selected {iteration_metrics['n_selected']} samples")
        logger.info(f"  Fraud ratio: {iteration_metrics['fraud_ratio']:.3f}")
        logger.info(f"  Avg uncertainty: {iteration_metrics['avg_uncertainty']:.4f}")

        return iteration_metrics

    def run_experiment(self, X_unlabeled, y_unlabeled, n_iterations=5,
                      samples_per_iteration=50, **kwargs):
        """
        Run complete active learning experiment
        """
        logger.info(f"Starting active learning experiment:")
        logger.info(f"  Iterations: {n_iterations}")
        logger.info(f"  Samples per iteration: {samples_per_iteration}")
        logger.info(f"  Strategy: {self.strategy.strategy_name}")

        results = []

        for i in range(n_iterations):
            iteration_result = self.run_iteration(
                X_unlabeled, y_unlabeled, samples_per_iteration, **kwargs
            )
            results.append(iteration_result)

        # Calculate experiment summary
        total_labeled = sum([r['n_selected'] for r in results])
        avg_fraud_ratio = np.mean([r['fraud_ratio'] for r in results])
        baseline_fraud_ratio = np.mean(y_unlabeled)
        improvement = (avg_fraud_ratio / baseline_fraud_ratio - 1) * 100 if baseline_fraud_ratio > 0 else 0

        summary = {
            'total_labeled': total_labeled,
            'avg_fraud_ratio': avg_fraud_ratio,
            'baseline_fraud_ratio': baseline_fraud_ratio,
            'improvement_percent': improvement,
            'strategy_used': self.strategy.strategy_name
        }

        logger.info(f"Experiment completed:")
        logger.info(f"  Total samples labeled: {summary['total_labeled']}")
        logger.info(f"  Average fraud detection rate: {summary['avg_fraud_ratio']:.3f}")
        logger.info(f"  Improvement over random: {summary['improvement_percent']:.1f}%")

        return results, summary

    def get_selection_statistics(self):
        """Get statistics about sample selection"""
        if not self.experiment_history:
            return None

        fraud_ratios = [exp['fraud_ratio'] for exp in self.experiment_history]
        uncertainties = [exp['avg_uncertainty'] for exp in self.experiment_history]

        stats = {
            'mean_fraud_ratio': np.mean(fraud_ratios),
            'std_fraud_ratio': np.std(fraud_ratios),
            'mean_uncertainty': np.mean(uncertainties),
            'std_uncertainty': np.std(uncertainties),
            'total_iterations': len(self.experiment_history),
            'total_samples_labeled': sum([exp['n_selected'] for exp in self.experiment_history])
        }

        return stats


# Utility functions for active learning
def evaluate_selection_quality(selected_indices, true_labels, baseline_fraud_rate):
    """
    Evaluate quality of sample selection
    """
    selected_labels = true_labels[selected_indices]
    fraud_ratio_selected = np.mean(selected_labels)

    # Calculate lift over random selection
    lift = fraud_ratio_selected / baseline_fraud_rate if baseline_fraud_rate > 0 else 1

    # Calculate precision and recall for fraud detection
    n_fraud_selected = np.sum(selected_labels)
    n_total_fraud = np.sum(true_labels)

    precision = fraud_ratio_selected  # Fraud ratio in selection
    recall = n_fraud_selected / n_total_fraud if n_total_fraud > 0 else 0

    return {
        'fraud_ratio_selected': fraud_ratio_selected,
        'baseline_fraud_rate': baseline_fraud_rate,
        'lift': lift,
        'precision': precision,
        'recall': recall,
        'n_fraud_selected': n_fraud_selected,
        'n_total_selected': len(selected_indices)
    }


def compare_strategies(model, X_unlabeled, y_unlabeled, strategies=None,
                      n_iterations=5, samples_per_iteration=50):
    """
    Compare different active learning strategies
    """
    if strategies is None:
        strategies = ['uncertainty', 'diversity', 'qbc']

    results = {}

    for strategy_name in strategies:
        logger.info(f"Testing strategy: {strategy_name}")

        manager = ActiveLearningManager(model, strategy_name)
        strategy_results, strategy_summary = manager.run_experiment(
            X_unlabeled, y_unlabeled, n_iterations, samples_per_iteration
        )

        results[strategy_name] = {
            'results': strategy_results,
            'summary': strategy_summary,
            'statistics': manager.get_selection_statistics()
        }

    # Find best performing strategy
    best_strategy = max(results.keys(),
                       key=lambda x: results[x]['summary']['avg_fraud_ratio'])

    logger.info(f"Best performing strategy: {best_strategy}")

    return results, best_strategy


def simulate_annotation_cost(X_samples, cost_model='uniform', **cost_params):
    """
    Simulate annotation costs for different samples
    """
    n_samples = len(X_samples)

    if cost_model == 'uniform':
        return np.ones(n_samples)
    elif cost_model == 'complexity':
        # Cost based on feature complexity
        complexity = np.mean(np.abs(X_samples), axis=1)
        return 1 + complexity / np.max(complexity)
    elif cost_model == 'random':
        # Random costs
        return np.random.uniform(0.5, 2.0, n_samples)
    elif cost_model == 'expert_time':
        # Cost based on expert time requirements
        base_time = cost_params.get('base_time', 5)  # minutes
        complexity_factor = np.random.uniform(0.8, 1.5, n_samples)
        return base_time * complexity_factor
    else:
        raise ValueError(f"Unknown cost model: {cost_model}")


# Export main classes and functions
__all__ = [
    'ActiveLearningStrategy',
    'UncertaintyBasedStrategy',
    'DiversityBasedStrategy',
    'QBCStrategy',
    'AdaptiveStrategy',
    'CostSensitiveStrategy',
    'ActiveLearningManager',
    'evaluate_selection_quality',
    'compare_strategies',
    'simulate_annotation_cost'
]