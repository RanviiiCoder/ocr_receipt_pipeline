from typing import List, Dict, Any

def generate_summary(receipts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates data across all processed receipts.
    Computes total spend, number of transactions, and spend per store.
    """
    total_spend = 0.0
    transactions = len(receipts)
    spend_per_store = {}
    
    for receipt in receipts:
        # Extract amount
        amount_str = receipt.get("total_amount", {}).get("value")
        amount = 0.0
        if amount_str is not None:
            try:
                amount = float(amount_str)
            except ValueError:
                pass
                
        total_spend += amount
        
        # Extract store
        store_str = receipt.get("store_name", {}).get("value")
        store = store_str if store_str is not None else "Unknown Store"
        
        if store not in spend_per_store:
            spend_per_store[store] = 0.0
        spend_per_store[store] += amount
        
    return {
        "total_spend": round(total_spend, 2),
        "number_of_transactions": transactions,
        "spend_per_store": {k: round(v, 2) for k, v in spend_per_store.items()}
    }
