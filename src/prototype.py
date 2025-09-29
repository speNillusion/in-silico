"""
AVISO DE SEGURANÇA: Este projeto é apenas computacional. Não contém instruções de laboratório nem protocolos para manipulação de organismos. Não tente realizar procedimentos de cultura, manipulação ou liberação de microrganismos com base neste código. Consulte sempre especialistas e normas de biossegurança.
"""

import os
import logging
from typing import List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from PIL import Image
import argparse
from dotenv import load_dotenv
from collections import Counter

try:
    import google.generativeai as genai
except ImportError:
    genai = None

load_dotenv()

logging.basicConfig(level=logging.INFO)

USE_MOCKS = os.getenv("USE_MOCKS", "false").lower() == "true" or os.getenv("CI", "false").lower() == "true"

class GeminiClient:
    """Client for interacting with the Gemini API."""

    def __init__(self) -> None:
        self.api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
        if self.api_key and genai:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")  # Supports vision
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
        return f"Mock analysis for images: {', '.join(valid_paths)}"

def analyze_plastic_degradation(image_paths: List[str]) -> str:
    """Analyze images for plastic degradation.

    Args:
        image_paths: List of image paths.

    Returns:
        Analysis result.
    """
    client = GeminiClient()
    prompt = (
        "Analyze these sequential images for plastic degradation over time. "
        "Provide sections: evaluation, estimated degradation rate (if possible), "
        "visible changes, and computational recommendations. "
        "Cite public sources where appropriate (placeholders for references)."
    )
    return client.analyze_images(image_paths, prompt)

class Pipeline:
    """Pipeline for processing and analyzing candidates."""

    def ingest_sources(self, dataset_path: Optional[str] = None) -> pd.DataFrame:
        """Ingest data from sources.

        Args:
            dataset_path: Optional path to CSV file.

        Returns:
            DataFrame with candidates.
        """
        if dataset_path and os.path.exists(dataset_path):
            return pd.read_csv(dataset_path)
        # Placeholder for real API calls (e.g., UniProt, NASA, PDB, AlphaFold)
        # Insert API calls here, e.g., requests.get('https://api.uniprot.org/...')
        logging.info("Using synthetic data as placeholder.")
        return pd.DataFrame({
            "id": range(10),
            "sequence": ["ACDEFGHIKLMNPQRSTVWY" * 5 for _ in range(10)],
            "env_temp": np.random.uniform(20, 60, 10),
        })

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from the dataframe.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with added features.
        """
        if "sequence" in df.columns:
            df["length"] = df["sequence"].str.len()
            def aa_composition(seq: str) -> dict[str, float]:
                total = len(seq)
                return {aa: count / total for aa, count in Counter(seq).items()}
            comp_df = pd.DataFrame(df["sequence"].apply(aa_composition).tolist()).fillna(0)
            df = pd.concat([df, comp_df], axis=1)
        if "env_temp" in df.columns:
            df["env_temp_mean"] = df["env_temp"]
        return df

    def train_surrogate(self, features: pd.DataFrame, labels: pd.Series, use_mock: bool = False) -> Any:
        """Train a surrogate model.

        Args:
            features: Feature DataFrame.
            labels: Target labels.
            use_mock: If True, return a mock model.

        Returns:
            Trained model or mock.
        """
        if use_mock or features.empty:
            class MockModel:
                def predict(self, X: pd.DataFrame) -> np.ndarray:
                    return np.random.uniform(0, 1, len(X))
            return MockModel()
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(features, labels)
        return model

    def predict_with_uncertainty(self, model: Any, X: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Predict with uncertainty estimation.

        Args:
            model: Trained model.
            X: Input features.

        Returns:
            Mean predictions and standard deviations.
        """
        if hasattr(model, "predict") and hasattr(model, "estimators_"):
            tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
            mean = pd.Series(np.mean(tree_preds, axis=0))
            std = pd.Series(np.std(tree_preds, axis=0))
            return mean, std
        preds = pd.Series(model.predict(X))
        return preds, pd.Series([0.0] * len(preds))

    def acquisition_function(self, preds_mean: pd.Series, preds_std: pd.Series, k: int = 5) -> List[int]:
        """Acquisition function to select top candidates.

        Args:
            preds_mean: Mean predictions.
            preds_std: Standard deviations.
            k: Number of top candidates.

        Returns:
            List of top indices.
        """
        # Simple Expected Improvement approximation: mean + std
        ei = preds_mean + preds_std
        return ei.nlargest(k).index.tolist()

    def generate_report(self, candidates: pd.DataFrame, scores: pd.Series, out_path: str = "output") -> None:
        """Generate report and save to files.

        Args:
            candidates: DataFrame of candidates.
            scores: Scores for candidates.
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
            f.write("\n\n## Justifications\nPrioritized based on predicted performance and uncertainty.\n")
            f.write("References: [Placeholder for public sources]\n")

class AgentOrchestrator:
    """Orchestrates the pipeline execution."""

    def __init__(self) -> None:
        self.pipeline = Pipeline()
        self.logger = logging.getLogger(__name__)

    def run(self, dataset: Optional[str] = None) -> None:
        """Run the pipeline.

        Args:
            dataset: Optional dataset path.
        """
        df = self.pipeline.ingest_sources(dataset)
        features = self.pipeline.feature_engineering(df)
        # Assume dummy labels for demonstration (in real, use labeled data)
        labels = pd.Series(np.random.uniform(0, 1, len(features)))  # Placeholder
        model = self.pipeline.train_surrogate(features.drop(columns=["id"], errors="ignore"), labels)
        mean, std = self.pipeline.predict_with_uncertainty(model, features)
        top_ids = self.pipeline.acquisition_function(mean, std)
        top_df = df.iloc[top_ids]
        top_scores = mean.iloc[top_ids]
        self.pipeline.generate_report(top_df, top_scores)
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