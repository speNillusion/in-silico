"""Tests for the prototype pipeline."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import pytest
import json

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.prototype import Pipeline, GeminiClient, analyze_plastic_degradation

class TestPipeline(unittest.TestCase):
    """Test cases for the Pipeline class."""

    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = Pipeline()

    @patch('src.prototype.GeminiClient')
    def test_ingest_sources_synthetic(self, mock_client_class):
        """Test that ingest_sources uses Gemini to generate synthetic data."""
        mock_client = MagicMock()
        mock_response = json.dumps({
            "candidates": [
                {"id": 1, "sequence": "ACDEFG", "env_temp": 25.0},
                {"id": 2, "sequence": "HIJKLM", "env_temp": 30.0}
            ]
        })
        mock_client.generate_text.return_value = mock_response
        mock_client_class.return_value = mock_client

        df = self.pipeline.ingest_sources()
        self.assertGreater(len(df), 0)
        self.assertIn("id", df.columns)
        self.assertIn("sequence", df.columns)
        self.assertIn("env_temp", df.columns)

    @patch('src.prototype.GeminiClient')
    def test_feature_engineering_columns(self, mock_client_class):
        """Test that feature_engineering uses Gemini to add features."""
        mock_client = MagicMock()
        input_df = pd.DataFrame({
            "id": [1, 2],
            "sequence": ["ACDEFG", "HIJKLM"],
            "env_temp": [25.0, 30.0]
        })
        mock_response = json.dumps(input_df.assign(length=[6, 6], env_temp_mean=[27.5, 27.5]).to_dict(orient='records'))
        mock_client.generate_text.return_value = mock_response
        mock_client_class.return_value = mock_client

        result = self.pipeline.feature_engineering(input_df)
        self.assertIn("length", result.columns)
        self.assertIn("env_temp_mean", result.columns)

    @patch('src.prototype.GeminiClient')
    def test_prioritize_candidates(self, mock_client_class):
        """Test that prioritize_candidates uses Gemini to select top candidates."""
        mock_client = MagicMock()
        input_df = pd.DataFrame({
            "id": [1, 2, 3],
            "sequence": ["ACDEFG", "HIJKLM", "NPQRST"],
            "env_temp": [25.0, 30.0, 35.0],
            "length": [6, 6, 6],
            "env_temp_mean": [30.0, 30.0, 30.0]
        })
        mock_response = json.dumps({"top_ids": [2, 3], "top_scores": [0.8, 0.7]})
        mock_client.generate_text.return_value = mock_response
        mock_client_class.return_value = mock_client

        top_ids, top_scores = self.pipeline.prioritize_candidates(input_df)
        self.assertEqual(len(top_ids), 2)
        self.assertEqual(len(top_scores), 2)

    @patch('src.prototype.GeminiClient')
    def test_generate_report(self, mock_client_class):
        """Test that generate_report uses Gemini to create a report."""
        mock_client = MagicMock()
        input_df = pd.DataFrame({"id": [1], "sequence": ["ACDEFG"]})
        mock_client.generate_text.return_value = "Mock report content"
        mock_client_class.return_value = mock_client

        report = self.pipeline.generate_report(input_df)
        self.assertTrue(report.startswith("Mock report"))

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