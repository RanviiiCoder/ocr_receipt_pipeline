import easyocr
import numpy as np
from typing import List, Dict, Any

class OCREngine:
    def __init__(self):
        # Initialize EasyOCR for English, forcing CPU mode as requested
        self.reader = easyocr.Reader(['en'], gpu=False)

    def extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extracts text from a preprocessed image using EasyOCR.
        Returns a list of dictionaries with text, bbox, and confidence.
        """
        # EasyOCR readtext returns a list of tuples: (bbox, text, confidence)
        raw_results = self.reader.readtext(image)
        
        results = []
        for bbox, text, conf in raw_results:
            # bbox is a list of 4 points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            results.append({
                "text": text,
                "bbox": [[float(p[0]), float(p[1])] for p in bbox],
                "confidence": float(conf)
            })
            
        return results

    @staticmethod
    def get_full_text(results: List[Dict[str, Any]]) -> str:
        """
        Concatenates all text lines extracted from the image.
        """
        return "\n".join([item["text"] for item in results])

    @staticmethod
    def get_average_confidence(results: List[Dict[str, Any]]) -> float:
        """
        Calculates the average OCR confidence across all extracted lines.
        """
        if not results:
            return 0.0
        total_conf = sum(item["confidence"] for item in results)
        return total_conf / len(results)
