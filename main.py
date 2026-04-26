import os
import glob
import json
import argparse
import traceback

from preprocessor import preprocess_image
from ocr_engine import OCREngine
from extractor import extract_all
from structurer import structure_receipt_data
from summarizer import generate_summary

def process_receipts(input_dir: str, output_dir: str):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported image formats
    image_extensions = ('*.png', '*.jpg', '*.jpeg')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        # Also check uppercase extensions
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))
        
    if not image_paths:
        print(f"No images found in {input_dir}. Supported formats: .png, .jpg, .jpeg")
        return

    print(f"Found {len(image_paths)} images. Initializing OCR Engine (CPU mode)...")
    try:
        ocr_engine = OCREngine()
    except Exception as e:
        print(f"Failed to initialize OCR Engine: {e}")
        return

    all_structured_data = []

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        receipt_id = os.path.splitext(filename)[0]
        print(f"\nProcessing {filename}...")
        
        try:
            # 1. Preprocess
            print("  - Preprocessing image...")
            processed_img = preprocess_image(img_path)
            
            # 2. OCR Extraction
            print("  - Running OCR...")
            raw_ocr_results = ocr_engine.extract_text(processed_img)
            
            if not raw_ocr_results:
                raise ValueError("No text detected in the image.")
                
            # 3. Field Extraction
            print("  - Extracting fields...")
            extracted_fields = extract_all(raw_ocr_results)
            
            # 4. Structure Data
            print("  - Structuring output...")
            structured_data = structure_receipt_data(receipt_id, extracted_fields)
            
            # Save individual JSON
            output_file = os.path.join(output_dir, f"{receipt_id}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2)
                
            all_structured_data.append(structured_data)
            print(f"  -> Successfully saved to {output_file}")
            
        except Exception as e:
            print(f"  -> Error processing {filename}: {e}")
            traceback.print_exc()
            
    # Generate and save summary
    if all_structured_data:
        print("\nGenerating expense summary...")
        summary_data = generate_summary(all_structured_data)
        summary_file = os.path.join(output_dir, "expense_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Saved summary to {summary_file}")
    else:
        print("\nNo receipts were successfully processed. Summary not generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Receipt Extraction Pipeline")
    parser.add_argument("--input", required=True, help="Path to the folder containing receipt images.")
    parser.add_argument("--output", required=True, help="Path to the folder where JSON outputs will be saved.")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist.")
    else:
        process_receipts(args.input, args.output)
