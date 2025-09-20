"""
Hybrid Autoencoder Model
Combines traditional autoencoder with Loss Prediction Module
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class HybridAutoencoder(keras.Model):
    """
    Hybrid Autoencoder with integrated Loss Prediction Module
    Combines reconstruction learning with uncertainty estimation
    """
    
    def __init__(self, input_dim, encoding_dims=[14, 7], dropout_rate=0.1):
        super(HybridAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.dropout_rate = dropout_rate
        
        # Build encoder layers
        self.encoder_layers = []
        prev_dim = input_dim
        for dim in encoding_dims:
            self.encoder_layers.extend([
                layers.Dense(dim, activation='tanh'),
                layers.Dropout(dropout_rate)
            ])
            prev_dim = dim
            
        # Build decoder layers
        self.decoder_layers = []
        for dim in reversed(encoding_dims[:-1]):
            self.decoder_layers.extend([
                layers.Dense(dim, activation='tanh'),
                layers.Dropout(dropout_rate)
            ])
        self.decoder_layers.append(layers.Dense(input_dim, activation='linear'))
        
        # Loss Prediction Module
        from models.loss_prediction import LossPredictionModule
        self.lpm = LossPredictionModule()
        
    def encode(self, x, training=None):
        """Encode input to latent representation"""
        encoded = x
        for layer in self.encoder_layers:
            encoded = layer(encoded, training=training)
        return encoded
        
    def decode(self, encoded, training=None):
        """Decode latent representation to output"""
        decoded = encoded
        for layer in self.decoder_layers:
            decoded = layer(decoded, training=training)
        return decoded
        
    def call(self, x, training=None):
        """Forward pass through the hybrid autoencoder"""
        # Encode-decode process
        encoded = self.encode(x, training=training)
        decoded = self.decode(encoded, training=training)
        
        # Get loss prediction from LPM
        predicted_loss = self.lmp(encoded, training=training)
        
        return decoded, predicted_loss
    
    def get_reconstruction_error(self, x):
        """Calculate reconstruction error for input"""
        reconstructed, _ = self(x, training=False)
        mse = tf.reduce_mean(tf.square(x - reconstructed), axis=1)
        return mse
    
    def get_latent_representation(self, x):
        """Get latent representation for input"""
        return self.encode(x, training=False)
    
    def save_weights(self, filepath):
        """Save model weights"""
        super().save_weights(filepath)
        
    def load_weights(self, filepath):
        """Load model weights"""
        super().load_weights(filepath)
        
    def summary(self):
        """Print model summary"""
        print("=" * 50)
        print("HYBRID AUTOENCODER MODEL SUMMARY")
        print("=" * 50)
        print(f"Input dimension: {self.input_dim}")
        print(f"Encoding dimensions: {self.encoding_dims}")
        print(f"Dropout rate: {self.dropout_rate}")
        print(f"Total parameters: {self.count_params():,}")
        print("=" * 50)


class CustomTrainingLoop:
    """
    Custom training loop for hybrid autoencoder
    Handles combined loss from reconstruction and LPM
    """
    
    def __init__(self, model, optimizer=None, reconstruction_weight=1.0, lmp_weight=0.1):
        self.model = model
        self.optimizer = optimizer or keras.optimizers.Adam(learning_rate=0.001)
        self.reconstruction_weight = reconstruction_weight
        self.lmp_weight = lmp_weight
        
        # Metrics tracking
        self.train_loss_tracker = keras.metrics.Mean(name="train_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.lmp_loss_tracker = keras.metrics.Mean(name="lmp_loss")
    
    @tf.function
    def train_step(self, batch):
        """Single training step"""
        with tf.GradientTape() as tape:
            # Forward pass
            reconstructed, predicted_loss = self.model(batch, training=True)
            
            # Reconstruction loss (MSE)
            reconstruction_loss = tf.reduce_mean(tf.square(batch - reconstructed))
            
            # Loss prediction loss
            actual_loss = tf.reduce_mean(tf.square(batch - reconstructed), axis=1, keepdims=True)
            lmp_loss = tf.reduce_mean(tf.square(actual_loss - predicted_loss))
            
            # Combined loss
            total_loss = (self.reconstruction_weight * reconstruction_loss + 
                         self.lmp_weight * lmp_loss)
        
        # Compute gradients and update weights
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        self.train_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.lmp_loss_tracker.update_state(lmp_loss)
        
        return {
            "total_loss": self.train_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "lmp_loss": self.lmp_loss_tracker.result()
        }
    
    def train(self, dataset, epochs, verbose=True):
        """Train the model"""
        history = {
            "total_loss": [],
            "reconstruction_loss": [],
            "lmp_loss": []
        }
        
        for epoch in range(epochs):
            # Reset metrics
            self.train_loss_tracker.reset_states()
            self.reconstruction_loss_tracker.reset_states()
            self.lmp_loss_tracker.reset_states()
            
            # Train on batches
            for batch in dataset:
                metrics = self.train_step(batch)
            
            # Store history
            for key in history:
                history[key].append(float(metrics[key]))
            
            # Print progress
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"  Total Loss: {metrics['total_loss']:.6f}")
                print(f"  Reconstruction: {metrics['reconstruction_loss']:.6f}")
                print(f"  LPM Loss: {metrics['lmp_loss']:.6f}")
        
        return history


# Model factory functions
def create_hybrid_autoencoder(input_dim, config=None):
    """Factory function to create hybrid autoencoder with config"""
    if config is None:
        config = {
            'encoding_dims': [14, 7],
            'dropout_rate': 0.1
        }
    
    return HybridAutoencoder(
        input_dim=input_dim,
        encoding_dims=config.get('encoding_dims', [14, 7]),
        dropout_rate=config.get('dropout_rate', 0.1)
    )


def create_training_loop(model, config=None):
    """Factory function to create training loop with config"""
    if config is None:
        config = {
            'learning_rate': 0.001,
            'reconstruction_weight': 1.0,
            'lmp_weight': 0.1
        }
    
    optimizer = keras.optimizers.Adam(learning_rate=config.get('learning_rate', 0.001))
    
    return CustomTrainingLoop(
        model=model,
        optimizer=optimizer,
        reconstruction_weight=config.get('reconstruction_weight', 1.0),
        lmp_weight=config.get('lmp_weight', 0.1)
    )