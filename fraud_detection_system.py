"""
Hybrid Fraud Detection System - Main Integration File
Combines all components into a unified fraud detection system
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import yaml
import os
from datetime import datetime
from pathlib import Path

# Import custom modules
from models.hybrid_autoencoder import HybridAutoencoder, CustomTrainingLoop, create_hybrid_autoencoder
from models.active_learning import ActiveLearningManager
from utils.data_preprocessing import DataPreprocessor
from utils.visualization import FraudVisualization
from utils.evaluation import ModelEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HybridFraudDetectionSystem:
    """
    Complete Hybrid Fraud Detection System
    Integrates autoencoder, loss prediction module, and active learning
    """
    
    def __init__(self, config_path=None, random_state=42):
        """
        Initialize the hybrid fraud detection system
        
        Args:
            config_path: Path to configuration file
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.config = self._load_config(config_path)
        
        # Set random seeds
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        # Initialize components
        self.data_preprocessor = None
        self.model = None
        self.training_loop = None
        self.active_learning_manager = None
        self.evaluator = ModelEvaluator()
        self.visualizer = FraudVisualization()
        
        # Model state
        self.is_trained = False
        self.threshold = None
        self.scaler = None
        self.training_history = {}
        
        logger.info("Hybrid Fraud Detection System initialized")
    
    def _load_config(self, config_path):
        """Load configuration from file or use defaults"""
        default_config = {
            'model': {
                'autoencoder': {
                    'encoding_dims': [14, 7],
                    'dropout_rate': 0.1
                },
                'loss_prediction_module': {
                    'hidden_dims': [64, 32, 16],
                    'dropout_rate': 0.2
                },
                'training': {
                    'learning_rate': 0.001,
                    'reconstruction_weight': 1.0,
                    'lmp_weight': 0.1
                }
            },
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'validation_split': 0.2,
                'early_stopping_patience': 10
            },
            'active_learning': {
                'strategy': 'uncertainty',
                'samples_per_iteration': 50,
                'max_iterations': 5
            },
            'data': {
                'test_size': 0.2,
                'threshold_percentile': 95,
                'scale_features': True
            },
            'evaluation': {
                'cross_validation': False,
                'cv_folds': 5,
                'business_metrics': True
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            # Deep merge configurations
            default_config.update(user_config)
            logger.info(f"Configuration loaded from {config_path}")
        else:
            logger.info("Using default configuration")
        
        return default_config
    
    def load_data(self, data_path=None, use_synthetic=False, **data_params):
        """
        Load and preprocess data
        
        Args:
            data_path: Path to credit card dataset
            use_synthetic: Whether to use synthetic data
            **data_params: Additional parameters for data generation
        """
        logger.info("Loading and preprocessing data...")
        
        # Initialize data preprocessor
        self.data_preprocessor = DataPreprocessor(random_state=self.random_state)
        
        # Load data
        if data_path and os.path.exists(data_path):
            df = self.data_preprocessor.load_real_data(data_path)
        else:
            synthetic_params = {
                'n_normal': data_params.get('n_normal', 50000),
                'n_fraud': data_params.get('n_fraud', 1000),
                'n_features': data_params.get('n_features', 29),
                'complexity': data_params.get('complexity', 'medium')
            }
            df = self.data_preprocessor.generate_synthetic_data(**synthetic_params)
        
        # Preprocess data
        df_processed = self.data_preprocessor.preprocess_data(df)
        
        # Split data
        split_result = self.data_preprocessor.split_data(
            df_processed, 
            test_size=self.config['data']['test_size'],
            validation_split=self.config['training'].get('validation_split', 0) > 0
        )
        
        if len(split_result) == 4:  # train/test split
            self.X_train, self.X_test, self.y_train, self.y_test = split_result
            self.X_val, self.y_val = None, None
        else:  # train/val/test split
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = split_result
        
        # Prepare training data (only normal transactions for autoencoder)
        normal_indices = self.y_train == 0
        self.X_train_normal = self.X_train[normal_indices]
        
        logger.info(f"Data loaded successfully:")
        logger.info(f"  Training samples (normal only): {len(self.X_train_normal)}")
        logger.info(f"  Validation samples: {len(self.X_val) if self.X_val is not None else 0}")
        logger.info(f"  Test samples: {len(self.X_test)} ({np.sum(self.y_test)} fraud)")
        
        return df_processed

    def build_model(self, input_dim=None):
        """
        Build the hybrid autoencoder model

        Args:
            input_dim: Input dimension (auto-detected if None)
        """
        if input_dim is None:
            if hasattr(self, 'X_train_normal'):
                input_dim = self.X_train_normal.shape[1]
            else:
                raise ValueError("Input dimension not specified and no training data available")

        logger.info(f"Building hybrid autoencoder model (input_dim={input_dim})")

        # Create model
        self.model = create_hybrid_autoencoder(
            input_dim,
            self.config['model']['autoencoder']
        )

        # Get training configuration
        training_config = self.config['model']['training']

        # Create training loop with properly extracted parameters
        self.training_loop = CustomTrainingLoop(
            model=self.model,
            learning_rate=training_config.get('learning_rate', 0.001),
            reconstruction_weight=training_config.get('reconstruction_weight', 1.0),
            lmp_weight=training_config.get('lmp_weight', 0.1)
        )

        logger.info("Model built successfully")
        
    def train(self, epochs=None, batch_size=None, verbose=True):
        """
        Train the hybrid autoencoder model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Whether to print training progress
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if not hasattr(self, 'X_train_normal'):
            raise ValueError("Training data not loaded. Call load_data() first.")
        
        epochs = epochs or self.config['training']['epochs']
        batch_size = batch_size or self.config['training']['batch_size']
        
        logger.info(f"Training model for {epochs} epochs with batch size {batch_size}")
        
        # Prepare dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(self.X_train_normal.astype(np.float32))
        train_dataset = train_dataset.batch(batch_size).shuffle(1000)
        
        # Train model
        self.training_history = self.training_loop.train(
            train_dataset, epochs=epochs, verbose=verbose
        )
        
        self.is_trained = True
        logger.info("Model training completed")
        
        return self.training_history
    
    def determine_threshold(self, percentile=None):
        """
        Determine optimal threshold for fraud detection
        
        Args:
            percentile: Percentile of reconstruction errors to use as threshold
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        percentile = percentile or self.config['data']['threshold_percentile']
        
        logger.info(f"Determining threshold using {percentile}th percentile")
        
        # Get reconstruction errors for test data
        reconstructed, _ = self.model(self.X_test.astype(np.float32), training=False)
        reconstruction_errors = tf.reduce_mean(tf.square(self.X_test - reconstructed), axis=1)
        
        # Use normal transactions to set threshold
        normal_errors = reconstruction_errors[self.y_test == 0]
        self.threshold = float(np.percentile(normal_errors, percentile))
        
        logger.info(f"Threshold determined: {self.threshold:.6f}")
        
        return self.threshold
    
    def predict(self, X):
        """
        Make fraud predictions on new data
        
        Args:
            X: Input features
            
        Returns:
            predictions: Binary fraud predictions
            reconstruction_errors: Reconstruction errors
            predicted_losses: Predicted losses from LPM
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.threshold is None:
            raise ValueError("Threshold not determined. Call determine_threshold() first.")
        
        # Get model predictions
        X_tensor = tf.constant(X.astype(np.float32))
        reconstructed, predicted_losses = self.model(X_tensor, training=False)
        
        # Calculate reconstruction errors
        reconstruction_errors = tf.reduce_mean(tf.square(X_tensor - reconstructed), axis=1)
        
        # Make binary predictions
        predictions = (reconstruction_errors > self.threshold).numpy().astype(int)
        
        return predictions, reconstruction_errors.numpy(), predicted_losses.numpy()
    
    def evaluate_model(self, X_test=None, y_test=None, detailed=True):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features (uses internal test set if None)
            y_test: Test labels (uses internal test set if None)
            detailed: Whether to include detailed evaluation
            
        Returns:
            evaluation_results: Dictionary of evaluation metrics
        """
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
        
        logger.info("Evaluating model performance")
        
        # Make predictions
        predictions, reconstruction_errors, predicted_losses = self.predict(X_test)
        
        # Evaluate using ModelEvaluator
        evaluation_results = self.evaluator.evaluate_binary_classification(
            y_test, predictions, reconstruction_errors, detailed=detailed
        )
        
        # Add model-specific metrics
        evaluation_results.update({
            'threshold': self.threshold,
            'mean_reconstruction_error': float(np.mean(reconstruction_errors)),
            'std_reconstruction_error': float(np.std(reconstruction_errors)),
            'lpm_correlation': float(np.corrcoef(reconstruction_errors, predicted_losses.flatten())[0, 1])
        })
        
        logger.info(f"Model evaluation completed:")
        logger.info(f"  Accuracy: {evaluation_results['accuracy']:.4f}")
        logger.info(f"  F1-Score: {evaluation_results['f1_score']:.4f}")
        logger.info(f"  ROC AUC: {evaluation_results['roc_auc']:.4f}")
        
        return evaluation_results
    
    def run_active_learning_experiment(self, X_unlabeled=None, y_unlabeled=None, 
                                     n_iterations=None, samples_per_iteration=None):
        """
        Run active learning experiment
        
        Args:
            X_unlabeled: Unlabeled features (uses part of test set if None)
            y_unlabeled: Hidden labels for simulation (uses part of test set if None)
            n_iterations: Number of active learning iterations
            samples_per_iteration: Samples to select per iteration
            
        Returns:
            active_learning_results: Results from active learning experiment
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Use defaults from config
        n_iterations = n_iterations or self.config['active_learning']['max_iterations']
        samples_per_iteration = samples_per_iteration or self.config['active_learning']['samples_per_iteration']
        strategy = self.config['active_learning']['strategy']
        
        # Use part of test set if no unlabeled data provided
        if X_unlabeled is None:
            n_unlabeled = min(1000, len(self.X_test) // 2)
            X_unlabeled = self.X_test[:n_unlabeled]
            y_unlabeled = self.y_test[:n_unlabeled]
        
        logger.info(f"Running active learning experiment:")
        logger.info(f"  Strategy: {strategy}")
        logger.info(f"  Iterations: {n_iterations}")
        logger.info(f"  Samples per iteration: {samples_per_iteration}")
        logger.info(f"  Unlabeled pool size: {len(X_unlabeled)}")
        
        # Initialize active learning manager
        self.active_learning_manager = ActiveLearningManager(
            self.model, strategy_type=strategy
        )
        
        # Run experiment
        results, summary = self.active_learning_manager.run_experiment(
            X_unlabeled, y_unlabeled, n_iterations, samples_per_iteration
        )
        
        return results, summary
    
    def visualize_results(self, save_plots=False, plots_dir='plots'):
        """
        Create comprehensive visualizations of results
        
        Args:
            save_plots: Whether to save plots to files
            plots_dir: Directory to save plots
        """
        if not self.is_trained:
            logger.warning("Model not trained. Limited visualizations available.")
            return
        
        logger.info("Creating visualizations")
        
        if save_plots:
            os.makedirs(plots_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Training history
        if self.training_history:
            save_path = f"{plots_dir}/training_history_{timestamp}.png" if save_plots else None
            self.visualizer.plot_model_training_history(self.training_history, save_path)
        
        # Detection results
        if hasattr(self, 'X_test') and self.threshold is not None:
            predictions, errors, predicted_losses = self.predict(self.X_test)
            
            save_path = f"{plots_dir}/detection_results_{timestamp}.png" if save_plots else None
            self.visualizer.plot_fraud_detection_results(
                self.X_test, self.y_test, predictions, errors, 
                predicted_losses, self.threshold, save_path
            )
        
        logger.info("Visualizations completed")
    
    def save_model(self, model_path):
        """
        Save trained model to file
        
        Args:
            model_path: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Nothing to save.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model weights
        self.model.save_weights(model_path)
        
        # Save additional information
        model_info = {
            'threshold': float(self.threshold) if self.threshold else None,
            'config': self.config,
            'training_history': self.training_history,
            'model_architecture': {
                'input_dim': self.X_train_normal.shape[1] if hasattr(self, 'X_train_normal') else None,
                'encoding_dims': self.config['model']['autoencoder']['encoding_dims']
            },
            'save_timestamp': datetime.now().isoformat()
        }
        
        info_path = model_path.replace('.h5', '_info.json').replace('.weights', '_info.json')
        with open(info_path, 'w') as f:
            import json
            json.dump(model_info, f, indent=2, default=str)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Model info saved to {info_path}")
    
    def load_model(self, model_path):
        """
        Load trained model from file
        
        Args:
            model_path: Path to saved model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model info
        info_path = model_path.replace('.h5', '_info.json').replace('.weights', '_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                import json
                model_info = json.load(f)
            
            # Update config and threshold
            self.config = model_info.get('config', self.config)
            self.threshold = model_info.get('threshold')
            self.training_history = model_info.get('training_history', {})
            
            # Build model with saved architecture
            if 'model_architecture' in model_info:
                input_dim = model_info['model_architecture']['input_dim']
                if input_dim:
                    self.build_model(input_dim)
        else:
            logger.warning(f"Model info file not found: {info_path}")
            logger.warning("Model architecture information may be incomplete")
        
        # Load model weights
        if self.model is not None:
            self.model.load_weights(model_path)
            self.is_trained = True
            logger.info(f"Model loaded from {model_path}")
        else:
            raise ValueError("Cannot load model weights without model architecture")
    
    def generate_report(self, report_path=None):
        """
        Generate comprehensive system report
        
        Args:
            report_path: Path to save report
            
        Returns:
            report: Dictionary containing all system information
        """
        logger.info("Generating comprehensive system report")
        
        # Evaluate model if not done already
        evaluation_results = self.evaluate_model() if self.is_trained else {}
        
        # Compile report
        report = {
            'system_info': {
                'system_type': 'Hybrid Fraud Detection System',
                'version': '1.0.0',
                'creation_timestamp': datetime.now().isoformat(),
                'random_state': self.random_state
            },
            'configuration': self.config,
            'data_info': {
                'training_samples': len(self.X_train_normal) if hasattr(self, 'X_train_normal') else 0,
                'test_samples': len(self.X_test) if hasattr(self, 'X_test') else 0,
                'fraud_ratio': float(np.mean(self.y_test)) if hasattr(self, 'y_test') else 0,
                'feature_dimensions': self.X_train_normal.shape[1] if hasattr(self, 'X_train_normal') else 0
            },
            'model_info': {
                'is_trained': self.is_trained,
                'threshold': float(self.threshold) if self.threshold else None,
                'training_epochs': len(self.training_history.get('total_loss', [])),
                'final_training_loss': float(self.training_history['total_loss'][-1]) if self.training_history.get('total_loss') else None
            },
            'performance_metrics': evaluation_results,
            'training_history': self.training_history
        }
        
        # Add active learning results if available
        if hasattr(self, 'active_learning_manager') and self.active_learning_manager:
            al_stats = self.active_learning_manager.get_selection_statistics()
            if al_stats:
                report['active_learning'] = al_stats
        
        # Save report if path provided
        if report_path:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, 'w') as f:
                import json
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {report_path}")
        
        return report
    
    def run_complete_pipeline(self, data_path=None, use_synthetic=True, 
                            save_model_path=None, save_report_path=None, 
                            create_visualizations=True, **kwargs):
        """
        Run complete fraud detection pipeline
        
        Args:
            data_path: Path to data file
            use_synthetic: Whether to use synthetic data
            save_model_path: Path to save trained model
            save_report_path: Path to save evaluation report
            create_visualizations: Whether to create visualizations
            **kwargs: Additional parameters
            
        Returns:
            pipeline_results: Complete pipeline results
        """
        logger.info("Starting complete fraud detection pipeline")
        
        pipeline_results = {}
        
        try:
            # 1. Load and preprocess data
            logger.info("Step 1: Loading and preprocessing data")
            df = self.load_data(data_path, use_synthetic, **kwargs)
            pipeline_results['data_loaded'] = True
            pipeline_results['data_shape'] = df.shape
            
            # 2. Build model
            logger.info("Step 2: Building hybrid model")
            self.build_model()
            pipeline_results['model_built'] = True
            
            # 3. Train model
            logger.info("Step 3: Training model")
            training_history = self.train(
                epochs=kwargs.get('epochs', self.config['training']['epochs']),
                batch_size=kwargs.get('batch_size', self.config['training']['batch_size'])
            )
            pipeline_results['model_trained'] = True
            pipeline_results['training_history'] = training_history
            
            # 4. Determine threshold
            logger.info("Step 4: Determining optimal threshold")
            threshold = self.determine_threshold()
            pipeline_results['threshold_determined'] = True
            pipeline_results['threshold'] = threshold
            
            # 5. Evaluate model
            logger.info("Step 5: Evaluating model performance")
            evaluation_results = self.evaluate_model()
            pipeline_results['model_evaluated'] = True
            pipeline_results['evaluation_results'] = evaluation_results
            
            # 6. Run active learning experiment
            if kwargs.get('run_active_learning', True):
                logger.info("Step 6: Running active learning experiment")
                al_results, al_summary = self.run_active_learning_experiment()
                pipeline_results['active_learning_completed'] = True
                pipeline_results['active_learning_results'] = al_results
                pipeline_results['active_learning_summary'] = al_summary
            
            # 7. Create visualizations
            if create_visualizations:
                logger.info("Step 7: Creating visualizations")
                self.visualize_results(
                    save_plots=kwargs.get('save_plots', False),
                    plots_dir=kwargs.get('plots_dir', 'plots')
                )
                pipeline_results['visualizations_created'] = True
            
            # 8. Save model
            if save_model_path:
                logger.info("Step 8: Saving trained model")
                self.save_model(save_model_path)
                pipeline_results['model_saved'] = save_model_path
            
            # 9. Generate report
            logger.info("Step 9: Generating comprehensive report")
            report = self.generate_report(save_report_path)
            pipeline_results['report_generated'] = True
            pipeline_results['report'] = report
            
            # Pipeline summary
            pipeline_results['pipeline_completed'] = True
            pipeline_results['completion_timestamp'] = datetime.now().isoformat()
            
            logger.info("Complete fraud detection pipeline finished successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            pipeline_results['pipeline_completed'] = False
            pipeline_results['error'] = str(e)
            raise
        
        return pipeline_results


# Factory function for easy system creation
def create_fraud_detection_system(config_path=None, random_state=42):
    """
    Factory function to create a fraud detection system
    
    Args:
        config_path: Path to configuration file
        random_state: Random seed
        
    Returns:
        HybridFraudDetectionSystem instance
    """
    return HybridFraudDetectionSystem(config_path, random_state)


# Convenience function for quick experimentation
def quick_fraud_detection_experiment(data_path=None, epochs=50, save_results=True):
    """
    Run a quick fraud detection experiment with minimal configuration
    
    Args:
        data_path: Path to data (uses synthetic if None)
        epochs: Number of training epochs
        save_results: Whether to save results
        
    Returns:
        system: Trained fraud detection system
        results: Experiment results
    """
    logger.info("Running quick fraud detection experiment")
    
    # Create system
    system = create_fraud_detection_system()
    
    # Run pipeline
    results = system.run_complete_pipeline(
        data_path=data_path,
        use_synthetic=(data_path is None),
        epochs=epochs,
        save_model_path='models/quick_experiment_model.h5' if save_results else None,
        save_report_path='results/quick_experiment_report.json' if save_results else None,
        create_visualizations=True,
        save_plots=save_results,
        plots_dir='plots/quick_experiment'
    )
    
    logger.info("Quick experiment completed")
    return system, results


if __name__ == "__main__":
    # Example usage
    system, results = quick_fraud_detection_experiment(epochs=20)
    
    print("\n🎉 Quick Fraud Detection Experiment Completed!")
    print(f"📊 Final Accuracy: {results['evaluation_results']['accuracy']:.4f}")
    print(f"🎯 Final F1-Score: {results['evaluation_results']['f1_score']:.4f}")
    print(f"📈 ROC AUC: {results['evaluation_results']['roc_auc']:.4f}")
    
    if 'active_learning_summary' in results:
        al_summary = results['active_learning_summary']
        print(f"🎓 Active Learning Improvement: {al_summary['improvement_percent']:.1f}%")
        print(f"🏷️ Total Samples Labeled: {al_summary['total_labeled']}")


# Export main classes and functions
__all__ = [
    'HybridFraudDetectionSystem',
    'create_fraud_detection_system',
    'quick_fraud_detection_experiment'
]