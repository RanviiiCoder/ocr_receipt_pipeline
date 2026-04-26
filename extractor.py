import re
from dateutil import parser
from typing import List, Dict, Any, Tuple, Optional

# Skip-words for item matching
SKIP_WORDS = {
    "tax", "subtotal", "sub total", "tip", "cashier", "thank you", 
    "change", "due", "total", "amount", "visa", "mastercard", "cash",
    "card", "balance"
}

def extract_store_name(results: List[Dict[str, Any]]) -> Tuple[Optional[str], float]:
    """
    Extracts the store name by taking the first non-trivial OCR line.
    Returns (value, confidence).
    """
    for item in results:
        text = item["text"].strip()
        # Non-trivial: at least 3 chars, contains letters
        if len(text) >= 3 and any(c.isalpha() for c in text):
            # For store name, we use the OCR confidence as the base
            return text, item["confidence"]
    return None, 0.0

def extract_date(results: List[Dict[str, Any]]) -> Tuple[Optional[str], float]:
    """
    Extracts the date using dateutil fuzzy parsing on lines that match a date regex.
    """
    date_pattern = re.compile(r'\b(?:\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b', re.IGNORECASE)
    
    best_date = None
    best_conf = 0.0
    
    for item in results:
        text = item["text"]
        if date_pattern.search(text):
            try:
                # Attempt to parse
                parsed_date = parser.parse(text, fuzzy=True)
                # Output in ISO format YYYY-MM-DD
                formatted_date = parsed_date.strftime("%Y-%m-%d")
                
                # Confidence combines OCR conf and pattern match certainty
                conf = min(1.0, item["confidence"] + 0.1) 
                
                if conf > best_conf:
                    best_date = formatted_date
                    best_conf = conf
            except (ValueError, OverflowError):
                continue
                
    return best_date, best_conf

def extract_total_amount(results: List[Dict[str, Any]]) -> Tuple[Optional[float], float]:
    """
    Extracts total amount using keyword-anchored regex, falling back to the largest price.
    """
    price_pattern = re.compile(r'[$£€]?\s*(\d+\.\d{2})')
    keyword_pattern = re.compile(r'(?i)(total|amount due|grand total)')
    
    best_total = None
    best_conf = 0.0
    
    largest_price = 0.0
    largest_price_conf = 0.0
    
    for i, item in enumerate(results):
        text = item["text"]
        
        # Check for prices to track the largest price as fallback
        price_matches = price_pattern.findall(text)
        for pm in price_matches:
            val = float(pm)
            if val > largest_price:
                largest_price = val
                largest_price_conf = item["confidence"] * 0.8  # Penalty for fallback

        # Check for keywords
        if keyword_pattern.search(text):
            # Look for price in the same line
            if price_matches:
                val = float(price_matches[-1])
                conf = min(1.0, item["confidence"] + 0.2)
                if conf > best_conf:
                    best_total = val
                    best_conf = conf
            else:
                # Look in the next line
                if i + 1 < len(results):
                    next_text = results[i+1]["text"]
                    next_price_matches = price_pattern.findall(next_text)
                    if next_price_matches:
                        val = float(next_price_matches[-1])
                        conf = min(1.0, (item["confidence"] + results[i+1]["confidence"])/2 + 0.1)
                        if conf > best_conf:
                            best_total = val
                            best_conf = conf
                            
    if best_total is not None:
        return best_total, best_conf
    elif largest_price > 0:
        return largest_price, largest_price_conf
        
    return None, 0.0

def extract_items(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extracts item lists using regex for lines in the format 'item name ... price'.
    Filters out skip-words.
    Returns a list of dicts: {"name": str, "price": float, "confidence": float}
    """
    items = []
    # Pattern looks for text followed by some spacing/characters and then a price
    item_pattern = re.compile(r'^(.*?)\s+[$£€]?\s*(\d+\.\d{2})$')
    
    for item in results:
        text = item["text"].strip()
        
        # Filter out short lines
        if len(text) < 5:
            continue
            
        # Check for skip words
        if any(skip in text.lower() for skip in SKIP_WORDS):
            continue
            
        match = item_pattern.match(text)
        if match:
            name = match.group(1).strip()
            price_str = match.group(2)
            
            if name and len(name) >= 2:
                items.append({
                    "name": name,
                    "price": float(price_str),
                    "confidence": item["confidence"]
                })
                
    return items

def extract_all(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Runs all extraction heuristics and returns raw extracted data with confidences.
    """
    store_name, store_conf = extract_store_name(results)
    date_val, date_conf = extract_date(results)
    total_val, total_conf = extract_total_amount(results)
    items = extract_items(results)
    
    return {
        "store_name": {"value": store_name, "confidence": store_conf},
        "date": {"value": date_val, "confidence": date_conf},
        "total_amount": {"value": total_val, "confidence": total_conf},
        "items": items
    }
