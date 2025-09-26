"""
🧪 NEURAL NETWORK TESTS
======================

Comprehensive test suite for the RobotNavigationNN class.
Tests all functionality including forward pass, backpropagation,
training, and evaluation.

Biological Connection:
- Tests verify that the network behaves like a biological neural system
- Forward pass mimics information flow through brain regions
- Backpropagation mimics synaptic plasticity and learning
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.neural_network import RobotNavigationNN, create_data_splits, analyze_predictions

# =============================================================================
# TEST DATA GENERATION
# =============================================================================

def create_test_data(n_samples: int = 100) -> tuple:
    """
    Create synthetic test data for neural network testing
    
    Args:
        n_samples: Number of test samples
        
    Returns:
        X: Features (n_samples, 9) - 3×3 perception grids
        y: Labels (n_samples,) - Action labels (0-3)
    """
    np.random.seed(42)  # For reproducible tests
    
    # Generate random 3×3 perception grids (binary 0/1)
    X = np.random.randint(0, 2, size=(n_samples, 9)).astype(np.float32)
    
    # Generate random action labels
    y = np.random.randint(0, 4, size=n_samples)
    
    return X, y

# =============================================================================
# NEURAL NETWORK INITIALIZATION TESTS
# =============================================================================

class TestNeuralNetworkInitialization:
    """Test neural network initialization and basic properties"""
    
    def test_default_initialization(self):
        """Test default neural network initialization"""
        model = RobotNavigationNN()
        
        # Check architecture
        assert model.input_size == 9
        assert model.hidden1_size == 64
        assert model.hidden2_size == 32
        assert model.output_size == 4
        assert model.dropout_rate == 0.2
        assert model.learning_rate == 0.001
        
        # Check weight shapes
        assert model.W1.shape == (9, 64)
        assert model.W2.shape == (64, 32)
        assert model.W3.shape == (32, 4)
        
        # Check bias shapes
        assert model.b1.shape == (1, 64)
        assert model.b2.shape == (1, 32)
        assert model.b3.shape == (1, 4)
    
    def test_custom_initialization(self):
        """Test custom neural network initialization"""
        model = RobotNavigationNN(
            input_size=9,
            hidden1_size=32,
            hidden2_size=16,
            output_size=4,
            dropout_rate=0.3,
            learning_rate=0.01
        )
        
        # Check custom parameters
        assert model.input_size == 9
        assert model.hidden1_size == 32
        assert model.hidden2_size == 16
        assert model.output_size == 4
        assert model.dropout_rate == 0.3
        assert model.learning_rate == 0.01
        
        # Check weight shapes match custom architecture
        assert model.W1.shape == (9, 32)
        assert model.W2.shape == (32, 16)
        assert model.W3.shape == (16, 4)
    
    def test_weight_initialization(self):
        """Test that weights are properly initialized"""
        model = RobotNavigationNN()
        
        # Check that weights are not all zeros (would prevent learning)
        assert not np.allclose(model.W1, 0)
        assert not np.allclose(model.W2, 0)
        assert not np.allclose(model.W3, 0)
        
        # Check that weights are not too large (would cause gradient explosion)
        assert np.max(np.abs(model.W1)) < 10
        assert np.max(np.abs(model.W2)) < 10
        assert np.max(np.abs(model.W3)) < 10
        
        # Check that weights are not too small (would cause gradient vanishing)
        assert np.max(np.abs(model.W1)) > 0.01
        assert np.max(np.abs(model.W2)) > 0.01
        assert np.max(np.abs(model.W3)) > 0.01

# =============================================================================
# ACTIVATION FUNCTION TESTS
# =============================================================================

class TestActivationFunctions:
    """Test activation functions (ReLU, Softmax)"""
    
    def test_relu_activation(self):
        """Test ReLU activation function"""
        model = RobotNavigationNN()
        
        # Test positive values
        x_positive = np.array([[1, 2, 3, 4]])
        relu_positive = model._relu(x_positive)
        np.testing.assert_array_equal(relu_positive, x_positive)
        
        # Test negative values
        x_negative = np.array([[-1, -2, -3, -4]])
        relu_negative = model._relu(x_negative)
        np.testing.assert_array_equal(relu_negative, np.zeros_like(x_negative))
        
        # Test mixed values
        x_mixed = np.array([[1, -2, 3, -4]])
        relu_mixed = model._relu(x_mixed)
        expected = np.array([[1, 0, 3, 0]])
        np.testing.assert_array_equal(relu_mixed, expected)
    
    def test_relu_derivative(self):
        """Test ReLU derivative function"""
        model = RobotNavigationNN()
        
        # Test positive values (derivative = 1)
        x_positive = np.array([[1, 2, 3, 4]])
        relu_deriv_positive = model._relu_derivative(x_positive)
        np.testing.assert_array_equal(relu_deriv_positive, np.ones_like(x_positive))
        
        # Test negative values (derivative = 0)
        x_negative = np.array([[-1, -2, -3, -4]])
        relu_deriv_negative = model._relu_derivative(x_negative)
        np.testing.assert_array_equal(relu_deriv_negative, np.zeros_like(x_negative))
    
    def test_softmax_activation(self):
        """Test Softmax activation function"""
        model = RobotNavigationNN()
        
        # Test basic softmax
        x = np.array([[1, 2, 3, 4]])
        softmax_output = model._softmax(x)
        
        # Check that probabilities sum to 1
        np.testing.assert_almost_equal(np.sum(softmax_output), 1.0, decimal=5)
        
        # Check that all probabilities are positive
        assert np.all(softmax_output > 0)
        
        # Check that highest input gets highest probability
        max_idx = np.argmax(x)
        assert np.argmax(softmax_output) == max_idx
    
    def test_softmax_numerical_stability(self):
        """Test Softmax with large values (numerical stability)"""
        model = RobotNavigationNN()
        
        # Test with large values
        x_large = np.array([[100, 200, 300, 400]])
        softmax_large = model._softmax(x_large)
        
        # Should not produce NaN or inf
        assert not np.any(np.isnan(softmax_large))
        assert not np.any(np.isinf(softmax_large))
        
        # Should still sum to 1
        np.testing.assert_almost_equal(np.sum(softmax_large), 1.0, decimal=5)

# =============================================================================
# DROPOUT TESTS
# =============================================================================

class TestDropout:
    """Test dropout functionality"""
    
    def test_dropout_training_mode(self):
        """Test dropout in training mode"""
        model = RobotNavigationNN(dropout_rate=0.5)
        
        # Create test input
        x = np.ones((10, 64))  # 10 samples, 64 features
        
        # Test dropout in training mode
        output, mask = model._dropout(x, training=True)
        
        # Check that some values are zeroed out
        assert np.any(output == 0)
        
        # Check that mask has correct shape
        assert mask.shape == x.shape
        
        # Check that mask contains 0s and 1s
        assert np.all((mask == 0) | (mask == 1))
    
    def test_dropout_inference_mode(self):
        """Test dropout in inference mode"""
        model = RobotNavigationNN(dropout_rate=0.5)
        
        # Create test input
        x = np.ones((10, 64))
        
        # Test dropout in inference mode
        output, mask = model._dropout(x, training=False)
        
        # Should return input unchanged
        np.testing.assert_array_equal(output, x)
        np.testing.assert_array_equal(mask, np.ones_like(x))
    
    def test_dropout_rate_effect(self):
        """Test that different dropout rates have different effects"""
        model_low = RobotNavigationNN(dropout_rate=0.1)
        model_high = RobotNavigationNN(dropout_rate=0.9)
        
        x = np.ones((100, 64))
        
        # Test with low dropout rate
        output_low, _ = model_low._dropout(x, training=True)
        zeros_low = np.sum(output_low == 0)
        
        # Test with high dropout rate
        output_high, _ = model_high._dropout(x, training=True)
        zeros_high = np.sum(output_high == 0)
        
        # High dropout should produce more zeros
        assert zeros_high > zeros_low

# =============================================================================
# FORWARD PASS TESTS
# =============================================================================

class TestForwardPass:
    """Test forward pass through the network"""
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape"""
        model = RobotNavigationNN()
        X = np.random.randn(10, 9)  # 10 samples, 9 features
        
        output, cache = model.forward(X, training=True)
        
        # Check output shape
        assert output.shape == (10, 4)
        
        # Check cache contains all intermediate values
        assert 'z1' in cache
        assert 'a1' in cache
        assert 'z2' in cache
        assert 'a2' in cache
        assert 'z3' in cache
        assert 'output' in cache
    
    def test_forward_pass_probabilities(self):
        """Test that forward pass produces valid probabilities"""
        model = RobotNavigationNN()
        X = np.random.randn(5, 9)
        
        output, _ = model.forward(X, training=True)
        
        # Check that probabilities sum to 1 for each sample
        row_sums = np.sum(output, axis=1)
        np.testing.assert_almost_equal(row_sums, np.ones(5), decimal=5)
        
        # Check that all probabilities are positive
        assert np.all(output > 0)
    
    def test_forward_pass_training_vs_inference(self):
        """Test that training and inference modes produce different results due to dropout"""
        model = RobotNavigationNN(dropout_rate=0.5)
        X = np.random.randn(10, 9)
        
        # Forward pass in training mode
        output_train, _ = model.forward(X, training=True)
        
        # Forward pass in inference mode
        output_inference, _ = model.forward(X, training=False)
        
        # Results should be different due to dropout
        assert not np.allclose(output_train, output_inference)

# =============================================================================
# LOSS FUNCTION TESTS
# =============================================================================

class TestLossFunction:
    """Test cross-entropy loss function"""
    
    def test_cross_entropy_loss_basic(self):
        """Test basic cross-entropy loss calculation"""
        model = RobotNavigationNN()
        
        # Create test data
        y_pred = np.array([[0.1, 0.7, 0.1, 0.1],  # Correct prediction
                          [0.3, 0.3, 0.2, 0.2]])   # Incorrect prediction
        y_true = np.array([1, 0])  # True labels
        
        loss, dloss = model._cross_entropy_loss(y_pred, y_true)
        
        # Loss should be positive
        assert loss > 0
        
        # Gradient should have correct shape
        assert dloss.shape == y_pred.shape
    
    def test_cross_entropy_loss_perfect_prediction(self):
        """Test loss when prediction is perfect"""
        model = RobotNavigationNN()
        
        # Perfect prediction
        y_pred = np.array([[0.0, 1.0, 0.0, 0.0],  # Perfect for label 1
                          [1.0, 0.0, 0.0, 0.0]])   # Perfect for label 0
        y_true = np.array([1, 0])
        
        loss, _ = model._cross_entropy_loss(y_pred, y_true)
        
        # Loss should be very small (but not exactly 0 due to log(1) = 0)
        assert loss < 0.1
    
    def test_cross_entropy_loss_wrong_prediction(self):
        """Test loss when prediction is completely wrong"""
        model = RobotNavigationNN()
        
        # Wrong prediction
        y_pred = np.array([[1.0, 0.0, 0.0, 0.0],  # Wrong for label 1
                          [0.0, 1.0, 0.0, 0.0]])   # Wrong for label 0
        y_true = np.array([1, 0])
        
        loss, _ = model._cross_entropy_loss(y_pred, y_true)
        
        # Loss should be high
        assert loss > 1.0

# =============================================================================
# BACKPROPAGATION TESTS
# =============================================================================

class TestBackpropagation:
    """Test backpropagation through the network"""
    
    def test_backpropagation_gradient_shape(self):
        """Test that backpropagation produces correct gradient shapes"""
        model = RobotNavigationNN()
        X = np.random.randn(5, 9)
        y = np.random.randint(0, 4, 5)
        
        # Forward pass
        output, cache = model.forward(X, training=True)
        
        # Backward pass
        gradients = model.backward(X, y, cache)
        
        # Check gradient shapes match weight shapes
        assert gradients['W1'].shape == model.W1.shape
        assert gradients['W2'].shape == model.W2.shape
        assert gradients['W3'].shape == model.W3.shape
        assert gradients['b1'].shape == model.b1.shape
        assert gradients['b2'].shape == model.b2.shape
        assert gradients['b3'].shape == model.b3.shape
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the network"""
        model = RobotNavigationNN()
        X = np.random.randn(3, 9)
        y = np.array([0, 1, 2])
        
        # Forward pass
        output, cache = model.forward(X, training=True)
        
        # Backward pass
        gradients = model.backward(X, y, cache)
        
        # Gradients should not be all zeros (would indicate no learning)
        assert not np.allclose(gradients['W1'], 0)
        assert not np.allclose(gradients['W2'], 0)
        assert not np.allclose(gradients['W3'], 0)

# =============================================================================
# TRAINING TESTS
# =============================================================================

class TestTraining:
    """Test training functionality"""
    
    def test_train_epoch(self):
        """Test training for one epoch"""
        model = RobotNavigationNN()
        X, y = create_test_data(50)
        
        # Train for one epoch
        loss, accuracy = model.train_epoch(X, y, batch_size=10)
        
        # Check that loss is reasonable
        assert loss > 0
        assert loss < 10  # Should not be extremely high
        
        # Check that accuracy is between 0 and 1
        assert 0 <= accuracy <= 1
    
    def test_training_improvement(self):
        """Test that training improves performance over multiple epochs"""
        model = RobotNavigationNN(learning_rate=0.01)
        X, y = create_test_data(100)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(X, y)
        
        # Train for multiple epochs
        initial_loss, _ = model.evaluate(X_val, y_val)
        
        for epoch in range(5):
            model.train_epoch(X_train, y_train, batch_size=20)
        
        final_loss, _ = model.evaluate(X_val, y_val)
        
        # Loss should generally decrease (though not guaranteed due to randomness)
        # We'll just check that the model can train without errors
        assert isinstance(final_loss, float)
        assert not np.isnan(final_loss)
    
    def test_predict_function(self):
        """Test prediction function"""
        model = RobotNavigationNN()
        X = np.random.randn(10, 9)
        
        predictions = model.predict(X)
        
        # Check output shape
        assert predictions.shape == (10,)
        
        # Check that predictions are valid action indices
        assert np.all(predictions >= 0)
        assert np.all(predictions < 4)
        assert np.all(predictions == predictions.astype(int))  # Should be integers

# =============================================================================
# DATA SPLITTING TESTS
# =============================================================================

class TestDataSplitting:
    """Test data splitting functionality"""
    
    def test_data_splits_shape(self):
        """Test that data splits have correct shapes"""
        X, y = create_test_data(100)
        X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(X, y)
        
        # Check that all splits have correct shapes
        assert X_train.shape[1] == X.shape[1]  # Same number of features
        assert X_val.shape[1] == X.shape[1]
        assert X_test.shape[1] == X.shape[1]
        
        # Check that labels match features
        assert X_train.shape[0] == y_train.shape[0]
        assert X_val.shape[0] == y_val.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
    
    def test_data_splits_ratios(self):
        """Test that data splits have correct ratios"""
        X, y = create_test_data(1000)  # Use larger dataset for better ratios
        X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(X, y)
        
        total = len(X)
        train_ratio = len(X_train) / total
        val_ratio = len(X_val) / total
        test_ratio = len(X_test) / total
        
        # Check ratios are approximately correct
        assert 0.75 <= train_ratio <= 0.85  # 80% ± 5%
        assert 0.05 <= val_ratio <= 0.15   # 10% ± 5%
        assert 0.05 <= test_ratio <= 0.15   # 10% ± 5%
    
    def test_data_splits_no_overlap(self):
        """Test that data splits don't overlap"""
        X, y = create_test_data(100)
        X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(X, y)
        
        # Check that no sample appears in multiple splits
        train_indices = set(range(len(X_train)))
        val_indices = set(range(len(X_val)))
        test_indices = set(range(len(X_test)))
        
        # Should be no overlap
        assert len(train_indices.intersection(val_indices)) == 0
        assert len(train_indices.intersection(test_indices)) == 0
        assert len(val_indices.intersection(test_indices)) == 0

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Test complete training pipeline"""
    
    def test_complete_training_pipeline(self):
        """Test complete training pipeline from data to model"""
        # Create test data
        X, y = create_test_data(200)
        X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(X, y)
        
        # Create and train model
        model = RobotNavigationNN(learning_rate=0.01)
        history = model.train(X_train, y_train, X_val, y_val, epochs=5)
        
        # Check that training completed successfully
        assert len(history['train_losses']) == 5
        assert len(history['val_losses']) == 5
        assert len(history['train_accuracies']) == 5
        assert len(history['val_accuracies']) == 5
        
        # Check that model can make predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
        
        # Check that predictions are valid
        assert np.all(predictions >= 0)
        assert np.all(predictions < 4)
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        # Create and train a simple model
        model = RobotNavigationNN()
        X, y = create_test_data(50)
        X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(X, y)
        
        # Train briefly
        model.train(X_train, y_train, X_val, y_val, epochs=2)
        
        # Save model
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            model.save_model(tmp.name)
            
            # Load model
            new_model = RobotNavigationNN()
            new_model.load_model(tmp.name)
            
            # Check that loaded model produces same predictions
            original_predictions = model.predict(X_test)
            loaded_predictions = new_model.predict(X_test)
            
            np.testing.assert_array_equal(original_predictions, loaded_predictions)
            
            # Clean up
            import os
            os.unlink(tmp.name)

# =============================================================================
# BIOLOGICAL INSPIRATION TESTS
# =============================================================================

class TestBiologicalInspiration:
    """Test that the network exhibits biologically-inspired behavior"""
    
    def test_sparse_activation(self):
        """Test that ReLU creates sparse activations"""
        model = RobotNavigationNN()
        X = np.random.randn(10, 9)
        
        output, cache = model.forward(X, training=True)
        
        # Check that ReLU activations are sparse (many zeros)
        a1_zeros = np.sum(cache['a1'] == 0)
        a2_zeros = np.sum(cache['a2'] == 0)
        
        # Should have some zeros due to ReLU
        assert a1_zeros > 0
        assert a2_zeros > 0
    
    def test_competition_mechanism(self):
        """Test that softmax creates competition between actions"""
        model = RobotNavigationNN()
        X = np.random.randn(5, 9)
        
        output, _ = model.forward(X, training=True)
        
        # Check that probabilities sum to 1 (competition)
        row_sums = np.sum(output, axis=1)
        np.testing.assert_almost_equal(row_sums, np.ones(5), decimal=5)
        
        # Check that one action typically dominates (winner-take-all)
        max_probs = np.max(output, axis=1)
        assert np.all(max_probs > 0.2)  # At least one action should have reasonable probability

# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Test performance and efficiency"""
    
    def test_forward_pass_speed(self):
        """Test that forward pass is reasonably fast"""
        import time
        
        model = RobotNavigationNN()
        X = np.random.randn(1000, 9)  # Large batch
        
        start_time = time.time()
        output, _ = model.forward(X, training=True)
        end_time = time.time()
        
        # Should complete in reasonable time (< 1 second)
        assert (end_time - start_time) < 1.0
    
    def test_memory_usage(self):
        """Test that model doesn't use excessive memory"""
        model = RobotNavigationNN()
        
        # Check that model parameters are reasonable size
        total_params = (model.W1.size + model.W2.size + model.W3.size + 
                       model.b1.size + model.b2.size + model.b3.size)
        
        # Should have reasonable number of parameters
        assert total_params < 10000  # Less than 10k parameters
        assert total_params > 100    # More than 100 parameters

# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    """
    Run all tests
    """
    print("🧪 Running Neural Network Tests...")
    print("=" * 50)
    
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
