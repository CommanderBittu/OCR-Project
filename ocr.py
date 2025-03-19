import os
import cv2
import numpy as np
import easyocr
import logging
import torch
from PIL import Image
from datetime import datetime, timezone

# Create necessary directories
os.makedirs('logs', exist_ok=True)
os.makedirs('debug', exist_ok=True)
os.makedirs('output', exist_ok=True)

# Set up logging with UTF-8 encoding for the StreamHandler
class UTF8StreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/ocr_log_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.log'),
        UTF8StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HindiTextRecognizer:
    def __init__(self, languages=['hi']):
        logger.info(f"Initializing Hindi Text Recognizer at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        try:
            # Initialize EasyOCR with basic settings
            self.reader = easyocr.Reader(
                lang_list=languages,
                gpu=True if torch.cuda.is_available() else False
            )
            logger.info("EasyOCR initialized successfully")

            if torch.cuda.is_available():
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
            else:
                logger.info("Using CPU for processing")

        except Exception as e:
            logger.error(f"Error initializing EasyOCR: {e}")
            raise

    def preprocess_image(self, image):
        logger.info("Starting image preprocessing")
        try:
            # Save original image
            cv2.imwrite(os.path.join("debug", "01_original.jpg"), image)

            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            cv2.imwrite(os.path.join("debug", "02_gray.jpg"), gray)

            # Enhance contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            cv2.imwrite(os.path.join("debug", "03_enhanced.jpg"), enhanced)

            # Denoise
            denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
            cv2.imwrite(os.path.join("debug", "04_denoised.jpg"), denoised)

            # Adaptive thresholding
            binary = cv2.adaptiveThreshold(
                denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                15,  # Block size
                5  # C constant
            )
            cv2.imwrite(os.path.join("debug", "05_binary.jpg"), binary)

            # Morphological operations
            kernel = np.ones((2, 2), np.uint8)
            morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            cv2.imwrite(os.path.join("debug", "06_morph.jpg"), morph)

            logger.info("Image preprocessing completed")
            return morph

        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise

    def detect_and_recognize_text(self, image_path):
        logger.info(f"Starting text detection for image: {image_path}")
        try:
            # Constants for text processing
            SIMILARITY_THRESHOLD = 0.85
            LINE_HEIGHT_FACTOR = 0.02
            CONFIDENCE_THRESHOLD = 0.3
            MAX_VERTICAL_MERGE_DISTANCE = 20

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Read original image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Get image dimensions and resize if too large
            height, width = img.shape[:2]
            logger.info(f"Original image dimensions: {width}x{height}")

            # Resize image if too large
            max_dimension = 2000
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height))
                height, width = img.shape[:2]
                logger.info(f"Resized image to: {new_width}x{new_height}")

            def normalize_text(text):
                """Normalize text by removing extra spaces and standardizing characters"""
                # Remove extra spaces and standardize spaces
                text = ' '.join(text.split())
                # Replace similar characters
                replacements = {
                    'ः': ':',
                    '़': '',
                    '॰': '.',
                    'ँ': 'ं',
                    '॥': '।',
                    'ॅ': '',
                    'ऍ': 'ए',
                    'ॉ': '',
                    '‍': '',
                    '‌': '',
                    'ई': 'इ',
                    'ओ': 'ो',
                    'े': 'े',
                    'ै': 'ै',
                    'ा': 'ा',
                    'ी': 'ी',
                    'ू': 'ू',
                    'ु': 'ु',
                    'ं': 'ं',
                    '्': '्',
                    'ृ': 'ृ'
                }
                for old, new in replacements.items():
                    text = text.replace(old, new)
                return text.strip()

            def text_similarity(text1, text2):
                """Calculate similarity between two texts using improved comparison"""
                text1 = normalize_text(text1)
                text2 = normalize_text(text2)

                if text1 == text2:
                    return 1.0

                if not text1 or not text2:
                    return 0.0

                # Convert texts to lists of words
                words1 = text1.split()
                words2 = text2.split()

                # If one text is completely contained within the other, consider them similar
                if text1 in text2 or text2 in text1:
                    return 0.9

                # Calculate word-based similarity
                common_words = set(words1) & set(words2)
                if not common_words:
                    return 0.0

                # Calculate overlap coefficient
                similarity = len(common_words) / min(len(set(words1)), len(set(words2)))

                return similarity

            def clean_hindi_text(text):
                """Clean and normalize Hindi text"""
                cleaned = ''
                for char in text:
                    if ('\u0900' <= char <= '\u097F' or  # Hindi range
                            char.isspace() or  # Spaces
                            char.isnumeric() or  # Numbers
                            char in '।॥॰'):  # Hindi punctuation
                        cleaned += char
                return normalize_text(cleaned)

            def merge_similar_segments(segments):
                """Merge similar text segments with improved word order preservation"""
                if not segments:
                    return []

                # Sort segments by x-coordinate (left to right)
                sorted_segments = sorted(segments, key=lambda x: x[0] if isinstance(x, tuple) else 0)
                merged = []

                for current in sorted_segments:
                    # Extract text from tuple if necessary
                    current_text = current[1] if isinstance(current, tuple) else current
                    current_text = normalize_text(current_text)

                    if not current_text:
                        continue

                    # Check if current segment should be merged with any existing segment
                    should_add = True
                    for i, existing in enumerate(merged):
                        similarity = text_similarity(current_text, existing)
                        if similarity > 0.6:
                            # If current is longer, replace existing
                            if len(current_text) > len(existing):
                                merged[i] = current_text
                            should_add = False
                            break

                    if should_add:
                        merged.append(current_text)

                return merged

            # Process OCR
            all_results = []

            # Direct OCR
            logger.info("Attempting direct OCR...")
            direct_results = self.reader.readtext(
                img,
                detail=1,
                paragraph=False,
                batch_size=8
            )
            all_results.extend(direct_results)

            # Preprocessed OCR
            logger.info("Attempting OCR with preprocessing...")
            processed_img = self.preprocess_image(img)
            processed_results = self.reader.readtext(
                processed_img,
                detail=1,
                paragraph=False,
                batch_size=8
            )
            all_results.extend(processed_results)

            # Sort results by position
            sorted_results = sorted(all_results,
                                    key=lambda x: (x[0][0][1], x[0][0][0]))

            # First pass: group by vertical position
            lines = []
            current_line_segments = []
            last_y = None
            line_height = height * LINE_HEIGHT_FACTOR

            for result in sorted_results:
                current_y = result[0][0][1]
                current_x = result[0][0][0]  # Get x-coordinate
                confidence = result[2] if len(result) > 2 else 0
                text = clean_hindi_text(result[1])

                if text and confidence > CONFIDENCE_THRESHOLD:
                    if last_y is None:
                        current_line_segments.append((current_x, text))  # Store x-coordinate with text
                    else:
                        vertical_distance = abs(current_y - last_y)
                        if vertical_distance <= line_height:
                            # Same line
                            current_line_segments.append((current_x, text))  # Store x-coordinate with text
                        else:
                            # New line
                            if current_line_segments:
                                # Sort segments within line by x-coordinate
                                current_line_segments.sort(key=lambda x: x[0])  # Sort by x-coordinate
                                merged_segments = merge_similar_segments(current_line_segments)
                                line_text = ' '.join(merged_segments)
                                if line_text.strip():
                                    lines.append(line_text)
                            current_line_segments = [(current_x, text)]  # Start new line with x-coordinate
                    last_y = current_y

            # Handle the last line
            if current_line_segments:
                # Sort segments within line by x-coordinate
                current_line_segments.sort(key=lambda x: x[0])  # Sort by x-coordinate
                merged_segments = merge_similar_segments(current_line_segments)
                line_text = ' '.join(merged_segments)
                if line_text.strip():
                    lines.append(line_text)

            # Second pass: remove duplicate lines
            unique_lines = []
            for line in lines:
                normalized = normalize_text(line)
                if not normalized:
                    continue

                # Check if this line is similar to any existing line
                should_add = True
                for existing in unique_lines:
                    if text_similarity(normalized, normalize_text(existing)) > SIMILARITY_THRESHOLD:
                        # If current line is longer, replace the existing one
                        if len(normalized) > len(normalize_text(existing)):
                            unique_lines[unique_lines.index(existing)] = line
                        should_add = False
                        break

                if should_add:
                    unique_lines.append(line)

            # Final cleaning pass
            final_lines = []
            for line in unique_lines:
                # Remove very short segments and clean up spaces
                words = [word for word in line.split() if len(word) > 1]
                cleaned = ' '.join(words)

                if cleaned and not any(text_similarity(cleaned, existing) > 0.8 for existing in final_lines):
                    final_lines.append(cleaned)

            # Create final formatted text with metadata
            metadata = [
            ]

            # Combine metadata and recognized text
            formatted_text = '\n'.join(metadata + final_lines)

            logger.debug(f"Final number of lines: {len(final_lines)}")
            if final_lines:
                logger.debug("Sample of detected text:")
                for i, line in enumerate(final_lines[:3]):
                    logger.debug(f"Line {i + 1}: {line}")

            return formatted_text.strip(), sorted_results, img

        except Exception as e:
            logger.error(f"Error during recognition: {e}")
            logger.error(f"Stack trace:", exc_info=True)
            raise

    def visualize_and_save_results(self, image_path, output_path="recognition_result.txt"):
        try:
            logger.info("Starting visualization and saving results")
            text, boxes, img = self.detect_and_recognize_text(image_path)

            if text and boxes is not None and img is not None:
                # Create visualization
                viz_img = img.copy()

                # Draw boxes and confidence scores
                for idx, box in enumerate(boxes):
                    # Convert points to proper format for drawing
                    points = np.array(box[0], np.int32).reshape((-1, 1, 2))

                    # Draw filled polygon with semi-transparent background
                    overlay = viz_img.copy()
                    cv2.fillPoly(overlay, [points], (255, 255, 200))
                    viz_img = cv2.addWeighted(overlay, 0.2, viz_img, 0.8, 0)

                    # Draw border
                    cv2.polylines(viz_img, [points], True, (0, 128, 0), 2)

                    # Add text confidence and index
                    if len(box) > 2:
                        conf = box[2]
                        if conf > 0.3:
                            # Get the top-left corner coordinates
                            x = int(box[0][0][0])
                            y = int(box[0][0][1])

                            # Draw text with proper coordinates
                            text_str = f'{idx + 1}:{conf:.2f}'
                            cv2.putText(
                                viz_img,
                                text_str,
                                (x, max(y - 5, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                1,
                                cv2.LINE_AA
                            )

                # Save results
                output_img_path = os.path.join("output", "annotated_result.jpg")
                cv2.imwrite(output_img_path, viz_img)

                # Save text with UTF-8 encoding
                output_text_path = os.path.join("output", output_path)
                with open(output_text_path, 'w', encoding='utf-8') as f:
                    f.write(text)

                # Print results
                print("\nRecognized Text:")
                print("-" * 50)
                print(text)
                print("-" * 50)
                print(f"\nResults saved to:")
                print(f"Text: {output_text_path}")
                print(f"Annotated image: {output_img_path}")

            return text

        except Exception as e:
            logger.error(f"Error in visualization: {e}")
            raise


def main():
    try:
        # Print script information
        current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        print(f"\nHindi OCR System")
        print(f"Current Date and Time (UTC): {current_time}")
        print(f"User: CommanderBittu")
        print("-" * 80)

        # Initialize recognizer
        recognizer = HindiTextRecognizer()

        # Define test image path
        test_image = r"C:/Users/shash/Downloads/a-latin-script-for-hindi-made-for-ease-of-writing-v0-pgfxx9up4yba1.jpg"

        if os.path.exists(test_image):
            print(f"\nProcessing image: {test_image}")
            recognized_text = recognizer.visualize_and_save_results(test_image)

            print("\nProcessing complete!")
            print("Check the 'output' directory for results and 'debug' directory for intermediate steps.")
        else:
            print(f"Error: Image not found at {test_image}")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\nAn error occurred: {e}")
        print("Check the log file for details.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()