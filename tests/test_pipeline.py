"""Tests for the prototype pipeline."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import pytest

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.prototype import Pipeline, GeminiClient, analyze_plastic_degradation

class TestPipeline(unittest.TestCase):
    """Test cases for the Pipeline class."""

    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = Pipeline()

    def test_ingest_sources_synthetic(self):
        """Test that ingest_sources creates synthetic data when no file is provided."""
        df = self.pipeline.ingest_sources()
        self.assertEqual(len(df), 10)
        self.assertIn("id", df.columns)
        self.assertIn("sequence", df.columns)
        self.assertIn("env_temp", df.columns)

    def test_feature_engineering_columns(self):
        """Test that feature_engineering adds the expected columns."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "sequence": ["ACDEFG", "HIJKLM", "NPQRST"],
            "env_temp": [25.0, 30.0, 35.0]
        })
        result = self.pipeline.feature_engineering(df)
        self.assertIn("length", result.columns)
        self.assertIn("env_temp_mean", result.columns)
        self.assertEqual(result["length"].tolist(), [6, 6, 6])

    def test_acquisition_function_returns_k_ids(self):
        """Test that acquisition_function returns exactly k IDs."""
        mean = pd.Series([0.1, 0.5, 0.3, 0.8, 0.2])
        std = pd.Series([0.05, 0.1, 0.2, 0.1, 0.3])
        k = 3
        result = self.pipeline.acquisition_function(mean, std, k)
        self.assertEqual(len(result), k)
        # Check that the highest values are selected (mean + std)
        expected_indices = [3, 4, 1]  # Indices of highest values
        self.assertEqual(set(result), set(expected_indices))

    def test_mock_model_prediction(self):
        """Test that the mock model returns predictions."""
        df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        model = self.pipeline.train_surrogate(df, pd.Series([0, 1, 0]), use_mock=True)
        mean, std = self.pipeline.predict_with_uncertainty(model, df)
        self.assertEqual(len(mean), 3)
        self.assertEqual(len(std), 3)


class TestGeminiClient(unittest.TestCase):
    """Test cases for the GeminiClient class."""

    @patch.dict(os.environ, {"USE_MOCKS": "true"})
    def test_generate_text_mock(self):
        """Test that generate_text returns a mock response when USE_MOCKS is true."""
        client = GeminiClient()
        response = client.generate_text("Test prompt")
        self.assertTrue(response.startswith("Mock text response"))

    @patch.dict(os.environ, {"USE_MOCKS": "true"})
    def test_analyze_images_mock(self):
        """Test that analyze_images returns a mock response when USE_MOCKS is true."""
        client = GeminiClient()
        response = client.analyze_images(["nonexistent.jpg"], "Test prompt")
        self.assertEqual(response, "No valid image paths provided.")


@patch("src.prototype.GeminiClient")
def test_analyze_plastic_degradation_with_nonexistent_images(mock_client_class):
    """Test that analyze_plastic_degradation handles nonexistent images gracefully."""
    # Setup mock
    mock_client = MagicMock()
    mock_client.analyze_images.return_value = "Mock analysis result"
    mock_client_class.return_value = mock_client

    # Call function with nonexistent images
    result = analyze_plastic_degradation(["nonexistent1.jpg", "nonexistent2.jpg"])
    
    # Verify mock was called
    mock_client.analyze_images.assert_called_once()
    
    # Verify result
    assert result == "Mock analysis result"

@patch.dict(os.environ, {"USE_MOCKS": "true"})
@patch("os.path.exists", return_value=True)
def test_analyze_plastic_degradation_mock_includes_type(mock_exists):
    """Test that analyze_plastic_degradation mock response includes plastic type."""
    result = analyze_plastic_degradation(["mock.jpg"])
    assert "Plastic Type" in result
    assert "Evaluation" in result


if __name__ == "__main__":
    pytest.main()