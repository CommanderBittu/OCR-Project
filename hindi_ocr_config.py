"""
EasyOCR Hindi Fine-tuning Configuration
Created: 2025-01-26 20:27:42 UTC
Author: CommanderBittu
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging
from datetime import datetime
import numpy as np
from tqdm import tqdm
import traceback
import random

# Create necessary directories
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('debug', exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            encoding='utf-8'
        ),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants and configurations
SCRIPT_CREATION_TIME = "2025-01-26 20:27:42"
SCRIPT_AUTHOR = "CommanderBittu"
TRAIN_ANNOTATIONS = "C:/Users/shash/Downloads/IIIT-HW-Hindi_v1/train.txt"
TRAIN_IMG_DIR = "C:/Users/shash/PycharmProjects/ocr/HindiSeg/train"
# Training safeguards
PATIENCE = 15
MAX_EPOCHS_WITHOUT_IMPROVEMENT = 10
MIN_LEARNING_RATE = 1e-6
EARLY_STOPPING_THRESHOLD = 0.001
VALIDATION_FREQUENCY = 1000  # Validate every N batches
SAVE_FREQUENCY = 5  # Save checkpoint every N epochs
CHECKPOINTS_DIR = 'checkpoints'

# Create checkpoints directory if it doesn't exist
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
# Image processing constants
DEFAULT_TARGET_HEIGHT = 32
MIN_TARGET_WIDTH = 32
MAX_TARGET_WIDTH = 512
DEFAULT_MEAN = 0.485
DEFAULT_STD = 0.229
DEVANAGARI_CHARS = set('अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहक्षत्रज्ञॐऍऑऎऔॅंः़्ािीुूृेैोौ्०१२३४५६७८९')
# Model configurations
NUM_CHANNELS = 1  # Grayscale images
HIDDEN_SIZE = 256
NUM_CLASSES = len(DEVANAGARI_CHARS) + 1  # +1 for CTC blank

# Training configurations
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
INITIAL_LR = 1e-4  # Reduced initial learning rate
FINAL_LR = 1e-5
NUM_EPOCHS = 100
PATIENCE = 15
WARMUP_EPOCHS = 3
GRADIENT_CLIP = 0.5
# Devanagari characters set

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            bidirectional=True,
            batch_first=True,
            dropout=0.5,
            num_layers=2
        )
        self.linear = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        output = self.linear(recurrent)
        return output

class FineTunedEasyOCR(nn.Module):
    def __init__(self, num_chars):
        super().__init__()
        logger.info("Initializing Fine-Tuned EasyOCR model")

        # CNN Feature Extraction with BatchNorm and Dropout
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Second conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Third conv block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.2),

            # Fourth conv block
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.2)
        )

        # Sequence modeling
        self.rnn = nn.Sequential(
            BidirectionalLSTM(1024, 512, 256),
            nn.Dropout(0.5),
            BidirectionalLSTM(256, 256, num_chars)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # CNN feature extraction
        conv = self.cnn(x)
        logger.debug(f"After CNN: {conv.shape}")

        # Reshape for sequence modeling
        b, c, h, w = conv.size()
        conv = conv.permute(0, 3, 1, 2)  # [batch, width, channel, height]
        conv = conv.reshape(b, w, c * h)  # [batch, width, features]

        logger.debug(f"Before RNN: {conv.shape}")
        output = self.rnn(conv)
        logger.debug(f"Final output: {output.shape}")

        return output


class GrayscaleConversion:
    def __call__(self, img):
        try:
            if not isinstance(img, Image.Image):
                raise ValueError("Input must be a PIL Image")
            return img.convert('L')
        except Exception as e:
            logger.error(f"Error in GrayscaleConversion: {str(e)}")
            raise e


class RandomAffineTransform:
    def __init__(self, degrees, translate, scale, shear, p=0.5):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.p = p

    def __call__(self, img):
        try:
            if not isinstance(img, Image.Image):
                raise ValueError("Input must be a PIL Image")

            if random.random() < self.p:
                return transforms.RandomAffine(
                    degrees=self.degrees,
                    translate=self.translate,
                    scale=self.scale,
                    shear=self.shear
                )(img)
            return img
        except Exception as e:
            logger.error(f"Error in RandomAffineTransform: {str(e)}")
            raise e
# Add these classes after FineTunedEasyOCR but before HindiOCRDataset
class HindiLabelProcessor:
    def __init__(self, char_set):
        self.chars = sorted(list(char_set))
        # Start from 1 because 0 is reserved for CTC blank
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(self.chars)}
        self.idx_to_char[0] = ''  # Blank label for CTC
        self.num_classes = len(self.chars) + 1  # +1 for blank

    def encode(self, text):
        """Convert text to sequence of indices"""
        return [self.char_to_idx[c] for c in text.strip() if c in self.char_to_idx]

    def decode(self, indices):
        """Convert sequence of indices to text, removing duplicates and blanks"""
        # Remove duplicates
        collapsed = []
        previous = None
        for idx in indices:
            if idx != previous:  # Remove duplicates
                collapsed.append(idx)
                previous = idx

        # Convert to text, ignoring blanks (0)
        return ''.join([self.idx_to_char.get(idx, '') for idx in collapsed if idx > 0])

    def decode_predictions(self, outputs):
        """Decode model output to text"""
        # outputs shape: [batch_size, sequence_length, num_classes]
        predictions = torch.argmax(outputs, dim=2)  # [batch_size, sequence_length]
        decoded_texts = []

        for pred in predictions:
            # Get the most likely character at each timestep
            pred = pred.cpu().numpy()

            # Remove duplicates and blanks
            collapsed = []
            previous = None
            for p in pred:
                if p != previous:  # Remove duplicates
                    collapsed.append(p)
                    previous = p

            # Convert to text, ignoring blanks (0)
            text = ''.join([self.idx_to_char.get(idx, '') for idx in collapsed if idx > 0])
            decoded_texts.append(text)

        return decoded_texts

class HindiImagePreprocessor:
    def __init__(self, target_height=32, min_width=32, max_width=512):
        self.target_height = target_height
        self.min_width = min_width
        self.max_width = max_width

    def __call__(self, image):
        try:
            # Ensure image is PIL Image
            if not isinstance(image, Image.Image):
                raise ValueError("Input must be a PIL Image")

            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')

            # Get dimensions
            width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image dimensions: {width}x{height}")

            # Calculate target width maintaining aspect ratio
            target_width = int(width * (self.target_height / height))
            target_width = max(self.min_width, min(self.max_width, target_width))

            # Resize image
            image = image.resize(
                (target_width, self.target_height),
                Image.Resampling.BILINEAR
            )

            # Convert to numpy array
            image_array = np.array(image, dtype=np.float32)

            # Normalize to [0, 1]
            image_array = image_array / 255.0

            # Convert to tensor
            tensor = torch.from_numpy(image_array)

            # Add channel dimension
            tensor = tensor.unsqueeze(0)

            return tensor

        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            raise e

class RandomAffineAugmentation:
    def __init__(self, p=0.5):
        self.p = p
        self.transform = transforms.RandomAffine(
            degrees=3,
            translate=(0.02, 0.02),
            scale=(0.95, 1.05),
            shear=2
        )

    def __call__(self, tensor):
        try:
            if random.random() < self.p:
                # Convert to PIL Image for affine transform
                image = transforms.ToPILImage()(tensor)
                transformed = self.transform(image)
                # Convert back to tensor
                return transforms.ToTensor()(transformed)
            return tensor
        except Exception as e:
            logger.error(f"Error in affine augmentation: {str(e)}")
            return tensor


class HindiOCRDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_dir = os.path.normpath(img_dir)
        self.transform = transform
        self.invalid_images = set()

        # Initialize characters first
        self.chars = sorted(list(DEVANAGARI_CHARS))

        # Initialize label processor
        self.label_processor = HindiLabelProcessor(DEVANAGARI_CHARS)

        # Create character mappings
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(self.chars)}
        self.idx_to_char[0] = ''  # Blank label for CTC

        # Load annotations with progress bar
        self.annotations = []
        logger.info("Loading annotations...")

        # First, count total lines
        total_lines = sum(1 for _ in open(annotations_file, 'r', encoding='utf-8'))

        # Now process with progress bar
        with open(annotations_file, 'r', encoding='utf-8') as file:
            pbar = tqdm(file, total=total_lines, desc="Processing annotations")
            batch_size = 1000  # Process in batches
            current_batch = []

            for line in pbar:
                if line.strip():
                    try:
                        img_path, text = line.strip().split(' ', 1)
                        img_path = img_path.replace('HindiSeg/train/', '').replace('\\', '/')
                        full_path = os.path.normpath(os.path.join(self.img_dir, img_path))
                        full_path = full_path.replace('\\', '/')

                        if os.path.exists(full_path):
                            current_batch.append((full_path, text))

                            # Process batch when it reaches the size
                            if len(current_batch) >= batch_size:
                                self._process_batch(current_batch)
                                current_batch = []

                    except Exception as e:
                        logger.debug(f"Error processing line: {line.strip()} - {str(e)}")

                    pbar.set_postfix({'valid_images': len(self.annotations)})

            # Process remaining items in the last batch
            if current_batch:
                self._process_batch(current_batch)

        logger.info(f"Found {len(self.annotations)} valid samples")

        # Validate a small sample
        self._validate_sample()

    def _process_batch(self, batch):
        """Process a batch of images in parallel"""
        valid_images = []
        for full_path, text in batch:
            try:
                # Quick validation without opening the image
                if os.path.getsize(full_path) > 0:
                    valid_images.append((full_path, text))
            except Exception as e:
                logger.debug(f"Error checking image {full_path}: {str(e)}")

        self.annotations.extend(valid_images)

    def _validate_sample(self, sample_size=10):
        """Validate a small random sample of images"""
        if not self.annotations:
            return

        logger.info("Validating sample images...")
        sample_indices = random.sample(range(len(self.annotations)), min(sample_size, len(self.annotations)))

        for idx in sample_indices:
            img_path, _ = self.annotations[idx]
            try:
                with Image.open(img_path) as img:
                    if img.size[0] <= 0 or img.size[1] <= 0:
                        self.invalid_images.add(img_path)
            except Exception as e:
                logger.debug(f"Invalid sample image {img_path}: {str(e)}")
                self.invalid_images.add(img_path)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        max_retries = 5
        current_idx = idx

        for attempt in range(max_retries):
            img_path, text = self.annotations[current_idx]

            if img_path in self.invalid_images:
                current_idx = (current_idx + 1) % len(self.annotations)
                continue

            try:
                # Load and process image
                with Image.open(img_path) as image:
                    if self.transform:
                        image_tensor = self.transform(image)
                    else:
                        preprocessor = HindiImagePreprocessor()
                        image_tensor = preprocessor(image)

                    # Encode text to label indices
                    label_indices = torch.tensor(self.label_processor.encode(text), dtype=torch.long)

                    return image_tensor, label_indices

            except Exception as e:
                logger.debug(f"Error processing {img_path}: {str(e)}")
                self.invalid_images.add(img_path)
                current_idx = (current_idx + 1) % len(self.annotations)
                continue

        # Return default values if all retries fail
        default_tensor = torch.zeros((1, 32, 32), dtype=torch.float32)
        default_label = torch.zeros(1, dtype=torch.long)
        return default_tensor, default_label

class HindiCollator:
    def __init__(self, label_processor):
        self.label_processor = label_processor

    def __call__(self, batch):
        # Filter None values
        batch = [item for item in batch if item is not None]
        if not batch:
            return None

        # Separate images and labels
        images, labels = zip(*batch)

        # Get max width for padding images
        max_width = max(img.size(-1) for img in images)

        # Pad images
        padded_images = []
        for img in images:
            padding = max_width - img.size(-1)
            if padding > 0:
                padded_img = torch.nn.functional.pad(img, (0, padding, 0, 0), "constant", 0)
            else:
                padded_img = img
            padded_images.append(padded_img)

        # Stack images
        images_tensor = torch.stack(padded_images)

        # Get label lengths
        label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
        max_label_length = label_lengths.max().item()

        # Pad labels
        padded_labels = torch.zeros(len(labels), max_label_length, dtype=torch.long)
        for i, label in enumerate(labels):
            padded_labels[i, :len(label)] = label

        return {
            'images': images_tensor,
            'labels': padded_labels,
            'label_lengths': label_lengths,
            'original_labels': labels  # Keep original labels for accuracy calculation
        }

# Training transforms
train_transform = transforms.Compose([
    HindiImagePreprocessor(
        target_height=DEFAULT_TARGET_HEIGHT,
        min_width=MIN_TARGET_WIDTH,
        max_width=MAX_TARGET_WIDTH
    ),
    RandomAffineAugmentation(p=0.5),
    transforms.Normalize(mean=[DEFAULT_MEAN], std=[DEFAULT_STD])
])

# Validation transform
val_transform = transforms.Compose([
    HindiImagePreprocessor(
        target_height=DEFAULT_TARGET_HEIGHT,
        min_width=MIN_TARGET_WIDTH,
        max_width=MAX_TARGET_WIDTH
    ),
    transforms.Normalize(mean=[DEFAULT_MEAN], std=[DEFAULT_STD])
])

logger.info(f"Configuration initialized by {SCRIPT_AUTHOR} at {SCRIPT_CREATION_TIME}")