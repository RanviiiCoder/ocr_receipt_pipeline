OCR Receipt Extraction Pipeline: Project Report
1. Approach
The objective of this project is to build a robust, production-ready pipeline for extracting structured data from receipt images. Given the variability in receipt formats, lighting conditions, and image angles, a multi-stage approach was adopted:
1.	Preprocessing: Before attempting OCR, the image undergoes several computer vision transformations to normalize its quality. This involves converting the image to grayscale, denoising it to remove artifacts, applying adaptive Gaussian thresholding to compensate for uneven lighting, and automatically detecting and correcting skew using affine rotations.
2.	OCR Engine: We utilize EasyOCR, a robust deep-learning-based OCR engine. To ensure compatibility and avoid GPU-dependency issues across different environments, the pipeline is explicitly configured to run in CPU mode.
3.	Information Extraction: Instead of relying solely on complex NLP models, we use a hybrid heuristic and regex-based approach. We anchor our searches to known structures (e.g., date formats, keywords like “Total” or “Amount Due”) and use confidence scoring that penalizes fallback mechanisms.
4.	Structuring & Summarization: The raw extracted data is standardized into a consistent JSON schema, flagging any fields where the calculated confidence falls below 0.7. Finally, an aggregation step computes overall metrics across all processed receipts.
2. Tools Used
Tool / Library	Purpose	Justification
Python	Core Language	Standard for data pipelines and ML tasks.
OpenCV (opencv-python)	Image Preprocessing	Industry standard for computer vision tasks like denoising, thresholding, and geometric transformations.
EasyOCR	Text Extraction	Excellent out-of-the-box accuracy for English text, provides bounding boxes and confidence scores natively.
NumPy	Data Manipulation	Required for efficient matrix operations on image data alongside OpenCV.
python-dateutil	Date Parsing	Provides robust “fuzzy” parsing capabilities to handle various date formats found on receipts.
3. Challenges Faced
1.	Variable Lighting and Contrast: Receipts are often crumpled or photographed in poor lighting. Solution: Implemented adaptive Gaussian thresholding which calculates threshold values locally for smaller regions of the image, rather than applying a global threshold.
2.	Skewed Images: Users rarely take perfectly straight photos. Solution: Utilized OpenCV’s minAreaRect to find the minimum bounding box of all text pixels and calculated the angle of rotation needed to deskew the image.
3.	Unpredictable Layouts for Totals: The “Total” amount can appear anywhere—sometimes inline with the word “Total”, sometimes on the next line, or sometimes without a label at all. Solution: Implemented a multi-tier fallback system. First, it looks for keyword anchors. If not found or if the keyword doesn’t have an associated price, it falls back to the largest currency value found on the receipt.
4.	Noise in Item Lists: OCR often picks up random characters or non-item lines (like “Thank you for shopping!”). Solution: Implemented a skip-word list (tax, subtotal, cashier, etc.) and enforced strict regex matching for the [Item Name] ... [Price] pattern.
4. Possible Improvements
•	Deep Learning for Layout Analysis: While regex and heuristics work for standard receipts, a LayoutLM model or a Graph Neural Network (GNN) could be trained to understand the spatial relationships between bounding boxes, significantly improving accuracy on complex or novel receipt layouts.
•	GPU Acceleration: Currently forced to CPU mode. Enabling CUDA/MPS support would dramatically decrease processing time per image, especially in batch-processing scenarios.
•	Dictionary Lookups: Integrating a spell-checker or a product-name database could help autocorrect OCR misreadings for item names.
•	Multi-language Support: Expanding the OCR engine initialization and the regex keyword dictionaries to support multiple languages.

