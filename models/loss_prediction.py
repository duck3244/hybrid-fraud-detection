"""
Loss Prediction Module (LPM)
Inspired by ELECTRA's discriminator for uncertainty estimation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class LossPredictionModule(keras.Model):
    """
    Loss Prediction Module inspired by ELECTRA's Active Learning approach
    Predicts the reconstruction loss for uncertainty-based sampling
    """
    
    def __init__(self, hidden_dims=[64, 32, 16], dropout_rate=0.2, activation='relu'):
        super(LossPredictionModule, self).__init__()
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.activation = activation
        
        # Build hidden layers
        self.hidden_layers = []
        for dim in hidden_dims:
            self.hidden_layers.extend([
                layers.Dense(dim, activation=activation),
                layers.Dropout(dropout_rate),
                layers.BatchNormalization()
            ])
        
        # Output layer (sigmoid for normalized loss prediction)
        self.output_layer = layers.Dense(1, activation='sigmoid')
        
        # Global average pooling for variable input shapes
        self.global_pool = layers.GlobalAveragePooling1D()
        
    def call(self, x, training=None):
        """Forward pass through LPM"""
        # Handle different input shapes
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=1)
        
        # Global average pooling
        x = self.global_pool(x)
        
        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        
        # Output prediction
        predicted_loss = self.output_layer(x)
        
        return predicted_loss
    
    def predict_uncertainty(self, encoded_features, training=False):
        """Predict uncertainty scores for encoded features"""
        predicted_losses = self(encoded_features, training=training)
        return predicted_losses.numpy().flatten()
    
    def get_config(self):
        """Get layer configuration"""
        return {
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation
        }


class AdaptiveLossPredictionModule(LossPredictionModule):
    """
    Adaptive LPM that adjusts its complexity based on input complexity
    """
    
    def __init__(self, input_dim, adaptive_scaling=True, **kwargs):
        # Adjust hidden dimensions based on input dimension
        if adaptive_scaling:
            base_dim = max(16, input_dim // 2)
            hidden_dims = [base_dim * 4, base_dim * 2, base_dim]
        else:
            hidden_dims = kwargs.get('hidden_dims', [64, 32, 16])
        
        super().__init__(hidden_dims=hidden_dims, **kwargs)
        self.input_dim = input_dim
        self.adaptive_scaling = adaptive_scaling
    
    def build(self, input_shape):
        """Build the model based on input shape"""
        super().build(input_shape)
        
        if self.adaptive_scaling:
            print(f"LPM adapted for input dim {self.input_dim}: {self.hidden_dims}")


class EnsembleLossPredictionModule(keras.Model):
    """
    Ensemble of multiple LPMs for improved uncertainty estimation
    """
    
    def __init__(self, n_models=3, **lmp_kwargs):
        super().__init__()
        self.n_models = n_models
        
        # Create ensemble of LPMs
        self.lmp_models = []
        for i in range(n_models):
            # Vary architecture slightly for diversity
            varied_kwargs = lmp_kwargs.copy()
            if 'hidden_dims' in varied_kwargs:
                base_dims = varied_kwargs['hidden_dims']
                # Add some variation to hidden dimensions
                variation = 0.8 + 0.4 * np.random.random()
                varied_kwargs['hidden_dims'] = [int(dim * variation) for dim in base_dims]
            
            self.lmp_models.append(LossPredictionModule(**varied_kwargs))
    
    def call(self, x, training=None):
        """Forward pass through ensemble"""
        predictions = []
        
        for model in self.lmp_models:
            pred = model(x, training=training)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = tf.reduce_mean(tf.stack(predictions, axis=0), axis=0)
        
        return ensemble_pred
    
    def predict_with_uncertainty(self, x, training=False):
        """Get predictions with uncertainty estimates"""
        predictions = []
        
        for model in self.lmp_models:
            pred = model(x, training=training)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        # Calculate mean and standard deviation
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred


class ContrastiveLossPredictionModule(LossPredictionModule):
    """
    LPM with contrastive learning for better uncertainty estimation
    """
    
    def __init__(self, temperature=0.1, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        
        # Additional projection head for contrastive learning
        self.projection_head = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32)  # Projection dimension
        ])
    
    def call(self, x, training=None):
        """Forward pass with contrastive learning"""
        # Standard LPM forward pass
        predicted_loss = super().call(x, training=training)
        
        if training:
            # Additional contrastive projection
            # Handle input shape for projection
            if len(x.shape) == 2:
                x = tf.expand_dims(x, axis=1)
            
            pooled_x = self.global_pool(x)
            projections = self.projection_head(pooled_x, training=training)
            
            return predicted_loss, projections
        else:
            return predicted_loss
    
    def contrastive_loss(self, projections, labels, temperature=None):
        """Calculate contrastive loss for better representation learning"""
        if temperature is None:
            temperature = self.temperature
        
        # Normalize projections
        projections = tf.nn.l2_normalize(projections, axis=1)
        
        # Calculate similarity matrix
        similarity_matrix = tf.matmul(projections, projections, transpose_b=True)
        similarity_matrix = similarity_matrix / temperature
        
        # Create mask for positive pairs (same labels)
        labels = tf.cast(labels, tf.float32)
        labels = tf.expand_dims(labels, 0)
        mask = tf.equal(labels, tf.transpose(labels))
        mask = tf.cast(mask, tf.float32)
        
        # Remove diagonal (self-similarity)
        mask = mask * (1 - tf.eye(tf.shape(projections)[0]))
        
        # Calculate contrastive loss
        exp_sim = tf.exp(similarity_matrix)
        sum_exp_sim = tf.reduce_sum(exp_sim, axis=1, keepdims=True)
        
        log_prob = similarity_matrix - tf.math.log(sum_exp_sim)
        mask_sum = tf.reduce_sum(mask, axis=1, keepdims=True)
        
        # Avoid division by zero
        mask_sum = tf.maximum(mask_sum, 1e-8)
        
        contrastive_loss = -tf.reduce_sum(mask * log_prob, axis=1) / mask_sum
        contrastive_loss = tf.reduce_mean(contrastive_loss)
        
        return contrastive_loss


# Factory functions
def create_lpm(lmp_type='standard', **kwargs):
    """Factory function to create different types of LPMs"""
    
    if lmp_type == 'standard':
        return LossPredictionModule(**kwargs)
    elif lmp_type == 'adaptive':
        return AdaptiveLossPredictionModule(**kwargs)
    elif lmp_type == 'ensemble':
        return EnsembleLossPredictionModule(**kwargs)
    elif lmp_type == 'contrastive':
        return ContrastiveLossPredictionModule(**kwargs)
    else:
        raise ValueError(f"Unknown LPM type: {lmp_type}")


def evaluate_lmp_performance(lmp, X_test, true_errors, threshold=0.5):
    """Evaluate LPM performance against true reconstruction errors"""
    
    # Get predictions
    if hasattr(lmp, 'predict_with_uncertainty'):
        predicted_losses, uncertainties = lmp.predict_with_uncertainty(X_test)
        predicted_losses = predicted_losses.flatten()
        uncertainties = uncertainties.flatten()
    else:
        predicted_losses = lmp.predict_uncertainty(X_test)
        uncertainties = None
    
    # Calculate metrics
    correlation = np.corrcoef(true_errors, predicted_losses)[0, 1]
    mae = np.mean(np.abs(true_errors - predicted_losses))
    mse = np.mean(np.square(true_errors - predicted_losses))
    
    # Classification metrics (if threshold is provided)
    if threshold is not None:
        true_anomalies = (true_errors > threshold).astype(int)
        pred_anomalies = (predicted_losses > threshold).astype(int)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        accuracy = accuracy_score(true_anomalies, pred_anomalies)
        precision = precision_score(true_anomalies, pred_anomalies, zero_division=0)
        recall = recall_score(true_anomalies, pred_anomalies, zero_division=0)
    else:
        accuracy = precision = recall = None
    
    results = {
        'correlation': correlation,
        'mae': mae,
        'mse': mse,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }
    
    if uncertainties is not None:
        results['mean_uncertainty'] = np.mean(uncertainties)
        results['std_uncertainty'] = np.std(uncertainties)
    
    return results