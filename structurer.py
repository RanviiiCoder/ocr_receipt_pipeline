from typing import Dict, Any

def _format_field(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper to format a single field dictionary.
    Adds the 'flagged' key based on confidence < 0.7.
    """
    # Use string representation for value if not None
    val = data.get("value")
    conf = data.get("confidence", 0.0)
    
    return {
        "value": str(val) if val is not None else None,
        "confidence": round(conf, 4),
        "flagged": conf < 0.7
    }

def structure_receipt_data(receipt_id: str, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes the raw extracted data and formats it into the final JSON specification.
    Adds the required confidence flags and aggregates low confidence fields.
    """
    structured = {
        "receipt_id": receipt_id,
        "store_name": _format_field(extracted_data["store_name"]),
        "date": _format_field(extracted_data["date"]),
        "total_amount": _format_field(extracted_data["total_amount"]),
        "items": [],
        "flags": {
            "low_confidence_fields": []
        }
    }
    
    # Process items
    for item in extracted_data["items"]:
        conf = item.get("confidence", 0.0)
        structured["items"].append({
            "name": str(item["name"]),
            "price": str(item["price"]),
            "confidence": round(conf, 4)
        })
        # Items are not explicitly listed in the main level 'flagged', but we can track if needed
        # We will track main fields in 'low_confidence_fields'
    
    # Populate flags for main fields
    for field_name in ["store_name", "date", "total_amount"]:
        if structured[field_name]["flagged"]:
            structured["flags"]["low_confidence_fields"].append(field_name)
            
    # Check if there are any low confidence items
    if any(item["confidence"] < 0.7 for item in structured["items"]):
        if "items" not in structured["flags"]["low_confidence_fields"]:
             structured["flags"]["low_confidence_fields"].append("items")
             
    return structured
