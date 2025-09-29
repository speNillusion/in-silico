"""
AVISO DE SEGURANÇA: Este projeto é apenas computacional. Não contém instruções de laboratório nem protocolos para manipulação de organismos. Não tente realizar procedimentos de cultura, manipulação ou liberação de microrganismos com base neste código. Consulte sempre especialistas e normas de biossegurança.
"""

import os
import logging
from typing import List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from PIL import Image
import argparse
from dotenv import load_dotenv
from collections import Counter

try:
    import google.generativeai as genai
except ImportError:
    genai = None

import json

load_dotenv()

logging.basicConfig(level=logging.INFO)

USE_MOCKS = os.getenv("USE_MOCKS", "false").lower() == "true" or os.getenv("CI", "false").lower() == "true"

class GeminiClient:
    """Client for interacting with the Gemini API."""

    def __init__(self) -> None:
        self.api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
        if self.api_key and genai:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-2.5-flash")  # Updated to valid model
        else:
            logging.warning("No API key or library found. Using mock mode.")
            self.model = None

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        """Generate text based on the given prompt.

        Args:
            prompt: The input prompt.
            **kwargs: Additional arguments for the model.

        Returns:
            Generated text or mock response.
        """
        if self.model and not USE_MOCKS:
            try:
                response = self.model.generate_content(prompt, **kwargs)
                return response.text
            except Exception as e:
                logging.error(f"Error generating text: {e}")
                return "Error in text generation."
        return f"Mock text response for prompt: {prompt[:50]}..."

    def analyze_images(self, image_paths: List[str], prompt: str) -> str:
        """Analyze images with the given prompt.

        Args:
            image_paths: List of image file paths.
            prompt: The analysis prompt.

        Returns:
            Analysis text or mock response.
        """
        valid_paths = [p for p in image_paths if os.path.exists(p)]
        if not valid_paths:
            return "No valid image paths provided."
        if self.model and not USE_MOCKS:
            try:
                images = [Image.open(p) for p in valid_paths]
                content = [prompt] + images
                response = self.model.generate_content(content)
                return response.text
            except Exception as e:
                logging.error(f"Error analyzing images: {e}")
                return "Error in image analysis."
        return (
            "Mock Analysis:\n"
            "Plastic Type: PET (Polyethylene Terephthalate)\n"
            "Evaluation: Moderate degradation observed.\n"
            "Estimated Degradation Rate: 5% per month.\n"
            "Visible Changes: Color fading and surface cracks.\n"
            "Computational Recommendations: Run further simulations with temperature variations.\n"
            "References: [Placeholder]"
        )

def analyze_plastic_degradation(image_paths: List[str]) -> str:
    """Analyze images for plastic degradation.

    Args:
        image_paths: List of image paths.

    Returns:
        Analysis result.
    """
    client = GeminiClient()
    prompt = (
        "First, identify and classify the type of plastic in the images (e.g., PET, HDPE, PVC, etc.). "
        "Then, analyze these sequential images for plastic degradation over time. "
        "Provide sections: plastic type, evaluation, estimated degradation rate (if possible), "
        "visible changes, and computational recommendations. "
        "Cite public sources where appropriate (placeholders for references)."
    )
    return client.analyze_images(image_paths, prompt)

class Pipeline:
    """Pipeline for processing and analyzing candidates."""

    def __init__(self) -> None:
        self.client = GeminiClient()

    def ingest_sources(self, dataset_path: Optional[str] = None) -> pd.DataFrame:
        """Ingest data from sources using AI if no file provided.

        Args:
            dataset_path: Optional path to CSV file.

        Returns:
            DataFrame with candidates.
        """
        if dataset_path and os.path.exists(dataset_path):
            return pd.read_csv(dataset_path)
        
        prompt = (
            "Generate 10 synthetic candidates for plastic degradation study. "
            "Each candidate should have 'id' (int), 'sequence' (str), 'env_temp' (float). "
            "Respond in JSON format as a list of dictionaries."
        )
        response = self.client.generate_text(prompt)
        
        try:
            data = json.loads(response)
            return pd.DataFrame(data)
        except (json.JSONDecodeError, Exception) as e:
            logging.error(f"Failed to parse AI response: {e}")
            # Fallback synthetic data
            return pd.DataFrame({
                "id": range(10),
                "sequence": ["ACDEFGHIKLMNPQRSTVWY" * 5 for _ in range(10)],
                "env_temp": np.random.uniform(20, 60, 10),
            })

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features using AI.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with added features.
        """
        df_json = df.to_json(orient='records')
        prompt = (
            f"Given this data: {df_json}, engineer additional features such as sequence length, "
            "amino acid composition, and any other relevant features for plastic degradation analysis. "
            "Respond in JSON format as a list of dictionaries with the updated data."
        )
        response = self.client.generate_text(prompt)
        
        try:
            updated_data = json.loads(response)
            return pd.DataFrame(updated_data)
        except (json.JSONDecodeError, Exception) as e:
            logging.error(f"Failed to parse AI response: {e}")
            # Fallback simple engineering
            if "sequence" in df.columns:
                df["length"] = df["sequence"].str.len()
            if "env_temp" in df.columns:
                df["env_temp_mean"] = df["env_temp"]
            return df

    def prioritize_candidates(self, df: pd.DataFrame, k: int = 5) -> Tuple[List[int], pd.Series, str]:
        """Prioritize candidates using AI analysis.

        Args:
            df: DataFrame with features.
            k: Number of top candidates.

        Returns:
            List of top indices, scores, and justifications string.
        """
        df_json = df.to_json(orient='records')
        prompt = (
            f"Analyze this data for plastic degradation potential: {df_json}. "
            f"Select the top {k} candidates, provide their ids (assuming 0-based index if not specified), "
            "scores (0-1 float), and a justification paragraph. "
            "Respond in JSON: {'top_indices': [int], 'scores': [float], 'justifications': str}"
        )
        response = self.client.generate_text(prompt)
        
        try:
            result = json.loads(response)
            top_ids = result['top_indices']
            scores = pd.Series(result['scores'])
            justifications = result['justifications']
            return top_ids, scores, justifications
        except (json.JSONDecodeError, KeyError, Exception) as e:
            logging.error(f"Failed to parse AI response: {e}")
            # Fallback dummy
            top_ids = list(range(min(k, len(df))))
            scores = pd.Series(np.random.uniform(0.5, 1.0, len(top_ids)))
            justifications = "Mock justifications: Prioritized based on simulated AI analysis."
            return top_ids, scores, justifications

    def generate_report(self, candidates: pd.DataFrame, scores: pd.Series, justifications: str, out_path: str = "output") -> None:
        """Generate report and save to files.

        Args:
            candidates: DataFrame of candidates.
            scores: Scores for candidates.
            justifications: Justification text from AI.
            out_path: Output directory.
        """
        os.makedirs(out_path, exist_ok=True)
        report_df = candidates.copy()
        report_df["score"] = scores
        report_df.to_csv(f"{out_path}/top_candidates.csv", index=False)
        with open(f"{out_path}/report.md", "w") as f:
            f.write("# Pipeline Report\n\n")
            f.write("## Top Candidates\n\n")
            f.write(report_df.to_markdown(index=False))
            f.write("\n\n## Justifications\n")
            f.write(justifications)
            f.write("\n\nReferences: [Placeholder for public sources]\n")

class AgentOrchestrator:
    """Orchestrates the pipeline execution."""

    def __init__(self) -> None:
        self.pipeline = Pipeline()
        self.logger = logging.getLogger(__name__)

    def run(self, dataset: Optional[str] = None) -> None:
        """Run the pipeline using AI.

        Args:
            dataset: Optional dataset path.
        """
        df = self.pipeline.ingest_sources(dataset)
        features = self.pipeline.feature_engineering(df)
        top_ids, scores, justifications = self.pipeline.prioritize_candidates(features)
        top_df = features.iloc[top_ids]
        self.pipeline.generate_report(top_df, scores, justifications)
        self.logger.info(f"Top candidates selected: {top_ids}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computational Prototype CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_sim = subparsers.add_parser("run-sim", help="Run simulation pipeline")
    run_sim.add_argument("--dataset", type=str, help="Path to dataset CSV (optional)")

    analyze = subparsers.add_parser("analyze-images", help="Analyze images for degradation")
    analyze.add_argument("images", nargs="*", help="Image paths")

    args = parser.parse_args()

    if args.command == "run-sim":
        orch = AgentOrchestrator()
        orch.run(args.dataset)
    elif args.command == "analyze-images":
        if not args.images:
            print("Please provide image paths.")
        else:
            result = analyze_plastic_degradation(args.images)
            print(result)
            os.makedirs("output", exist_ok=True)
            with open("output/analysis.txt", "w") as f:
                f.write(result)