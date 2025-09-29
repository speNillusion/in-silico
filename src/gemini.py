import os
import google.generativeai as genai
from typing import Optional, Literal, Any, List
from dataclasses import dataclass
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# ... (previous code remains the same until Gemini class) ...

class Gemini:
    def __init__(self) -> None:
        # ... (previous init code remains the same) ...
        
        # Initialize the model for vision tasks
        self.model = genai.GenerativeModel('gemini-pro-vision')
    
    def analyze_plastic_degradation(self, image_paths: List[str]) -> str:
        """
        Analyze a sequence of images showing plastic degradation by fungi
        
        Args:
            image_paths: List of paths to the images in chronological order
        
        Returns:
            str: Detailed analysis of the degradation process
        """
        try:
            images = []
            for path in image_paths:
                img = Image.open(path)
                images.append(img)
            
            prompt = """
            Analyze these images showing plastic degradation by fungi over time.
            Please provide:
            1. Assessment of degradation progress
            2. Degradation rate and efficiency
            3. Visible changes in the plastic structure
            4. Whether the fungal degradation appears to be working correctly
            5. Any recommendations or concerns
            """
            
            response = self.model.generate_content([prompt, *images])
            return response.text
            
        except Exception as e:
            return f"Error analyzing images: {str(e)}"
        

if __name__ == "__main__":
    gemini = Gemini()
    image_paths = [
        "path/to/image1.jpg",
        "path/to/image2.jpg",
        "path/to/image3.jpg"
    ]
    analysis = gemini.analyze_plastic_degradation(image_paths)
    print(analysis)
