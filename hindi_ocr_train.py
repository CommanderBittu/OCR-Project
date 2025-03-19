import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging
from tqdm import tqdm
import traceback
from hindi_ocr_config import *

def debug_predictions(decoded_preds, original_texts, max_samples=5):
    """Print debug information about predictions vs actual text"""
    samples = list(zip(decoded_preds, original_texts))[:max_samples]
    logger.debug("\nPrediction Samples:")
    for i, (pred, target) in enumerate(samples):
        logger.debug(f"\nSample {i + 1}:")
        logger.debug(f"Predicted: '{pred}'")
        logger.debug(f"Actual   : '{target}'")
        logger.debug(f"Length   : Pred={len(pred)}, Target={len(target)}")
        if len(pred) > 0 and len(target) > 0:
            match = sum(p == t for p, t in zip(pred[:min(len(pred), len(target))], target[:min(len(pred), len(target))]))
            logger.debug(f"Matches  : {match}/{min(len(pred), len(target))} characters")


def compute_accuracy(predictions, targets):
    """Compute character-level accuracy"""
    if not predictions or not targets:
        return 0.0

    correct_chars = 0
    total_chars = 0

    for pred, target in zip(predictions, targets):
        # Remove any whitespace
        pred = pred.strip()
        target = target.strip()

        # Log sample predictions occasionally
        if random.random() < 0.01:  # 1% chance to log
            logger.debug(f"\nPrediction: '{pred}'")
            logger.debug(f"Target    : '{target}'")

        # Calculate character-level accuracy
        for p_char, t_char in zip(pred, target):
            if p_char == t_char:
                correct_chars += 1
        total_chars += max(len(pred), len(target))

    return correct_chars / total_chars if total_chars > 0 else 0.0

def compute_cer(predictions, targets):
    """Compute Character Error Rate"""
    total_distance = 0
    total_length = 0

    for pred, target in zip(predictions, targets):
        # Calculate Levenshtein distance
        distance = levenshtein_distance(pred, target)
        total_distance += distance
        total_length += len(target)

    return total_distance / total_length if total_length > 0 else 1.0


def levenshtein_distance(s1, s2):
    """Compute the Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def decode_predictions(predictions, label_lengths, idx_to_char):
    """Decode CTC predictions to text."""
    texts = []
    for pred, length in zip(predictions, label_lengths):
        pred = pred[:length]
        text = []
        previous = None
        for p in pred:
            p = p.item()
            if p != previous and p != 0:  # Skip blank label and repeated characters
                text.append(idx_to_char.get(p, ''))
            previous = p
        texts.append(''.join(text))
    return texts

def validate_batch_sizes(loader, batch_size):
    """Validate that all batches meet size requirements."""
    for batch in loader:
        if batch is None:
            continue
        images, _, _ = batch
        if images.size(0) != batch_size:
            return False
    return True


def train_epoch(model, train_loader, criterion, optimizer, device, label_processor, epoch):
    model.train()
    total_loss = 0
    total_accuracy = 0
    total_cer = 0
    num_batches = 0

    # Initialize tracking variables
    min_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    loss_history = []  # Initialize loss history list

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")

    for batch_idx, batch in enumerate(progress_bar):
        if batch is None:
            continue

        try:
            # Move data to device
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            label_lengths = batch['label_lengths'].to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)  # [batch_size, seq_len, num_classes]
            batch_size = outputs.size(0)
            seq_length = outputs.size(1)

            # Calculate CTC loss
            log_probs = nn.functional.log_softmax(outputs, dim=2)
            input_lengths = torch.full(
                size=(batch_size,),
                fill_value=seq_length,
                dtype=torch.long,
                device=device
            )

            # Debug sizes
            logger.debug(f"Batch size: {batch_size}")
            logger.debug(f"Sequence length: {seq_length}")
            logger.debug(f"log_probs shape: {log_probs.shape}")
            logger.debug(f"labels shape: {labels.shape}")
            logger.debug(f"input_lengths shape: {input_lengths.shape}")
            logger.debug(f"label_lengths shape: {label_lengths.shape}")

            # Calculate loss
            loss = criterion(
                log_probs.transpose(0, 1),  # (T, N, C)
                labels,  # (N, S)
                input_lengths,  # (N,)
                label_lengths  # (N,)
            )

            # Track loss trend
            loss_history.append(loss.item())
            if len(loss_history) > 100:  # Track last 100 batches
                loss_trend = sum(loss_history[-10:]) / 10  # Moving average

                # Check for training issues
                if loss_trend > sum(loss_history[-100:-90]) / 10:
                    patience_counter += 1
                else:
                    patience_counter = 0

                # Save best model
                if loss_trend < min_loss:
                    min_loss = loss_trend
                    best_model_state = model.state_dict().copy()

                # Check for training issues
                if patience_counter >= PATIENCE:
                    logger.warning(f"Loss not improving for {PATIENCE} iterations. Reducing learning rate.")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5
                        if param_group['lr'] < 1e-6:
                            logger.info("Learning rate too low. Stopping training.")
                            return None, None, None
                    patience_counter = 0

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

            # Update weights
            optimizer.step()

            # Decode predictions for accuracy calculation
            with torch.no_grad():
                # Get predictions
                predictions = torch.argmax(outputs, dim=2)  # [batch_size, seq_len]

                # Decode predictions and targets
                decoded_preds = []
                original_texts = []

                for i in range(batch_size):
                    pred_indices = predictions[i].cpu().numpy()
                    target_length = label_lengths[i].item()
                    target_indices = labels[i][:target_length].cpu().numpy()

                    # Decode prediction
                    pred_text = label_processor.decode(pred_indices)
                    target_text = label_processor.decode(target_indices)

                    decoded_preds.append(pred_text)
                    original_texts.append(target_text)

                # Debug predictions occasionally
                if batch_idx % 100 == 0:
                    logger.info("\nSample Predictions:")
                    for pred, target in zip(decoded_preds[:3], original_texts[:3]):
                        logger.info(f"Pred : {pred}")
                        logger.info(f"Truth: {target}")
                        logger.info("-" * 50)

                # Calculate metrics
                accuracy = compute_accuracy(decoded_preds, original_texts)
                cer = compute_cer(decoded_preds, original_texts)

            # Update running metrics
            total_loss += loss.item()
            total_accuracy += accuracy
            total_cer += cer
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.4f}',
                'cer': f'{cer:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}',
                'avg_loss': f'{total_loss / num_batches:.4f}',
                'avg_acc': f'{total_accuracy / num_batches:.4f}'
            })

        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            logger.error(traceback.format_exc())
            continue

    # Save final model state if it's the best
    if best_model_state is not None:
        torch.save({
            'epoch': epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': min_loss,
        }, f'checkpoints/best_model_epoch_{epoch}.pth')

    return total_loss / num_batches, total_accuracy / num_batches, total_cer / num_batches
def validate_epoch(model, val_loader, criterion, device, label_processor):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_cer = 0
    num_batches = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")

        for batch_idx, batch in enumerate(progress_bar):
            if batch is None:
                continue

            try:
                # Move data to device
                images = batch['images'].to(device)
                labels = batch['labels'].to(device)
                label_lengths = batch['label_lengths'].to(device)

                # Forward pass
                outputs = model(images)

                # Calculate loss
                log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
                input_lengths = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long).to(device)
                loss = criterion(log_probs.transpose(0, 1), labels, input_lengths, label_lengths)

                # Decode predictions
                decoded_preds = label_processor.decode_predictions(outputs)
                original_texts = [label_processor.decode(label[:length].tolist())
                                  for label, length in zip(batch['labels'], batch['label_lengths'])]

                # Calculate metrics
                accuracy = compute_accuracy(decoded_preds, original_texts)
                cer = compute_cer(decoded_preds, original_texts)

                # Update running metrics
                total_loss += loss.item()
                total_accuracy += accuracy
                total_cer += cer
                num_batches += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'val_loss': f'{loss.item():.4f}',
                    'val_acc': f'{accuracy:.4f}',
                    'val_cer': f'{cer:.4f}'
                })

            except Exception as e:
                logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                logger.error(traceback.format_exc())
                continue

    # Return average metrics
    return (total_loss / num_batches,
            total_accuracy / num_batches,
            total_cer / num_batches)
def evaluate(model, loader, criterion, device, dataset, writer, epoch):
    """Evaluate the model on the validation set."""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_cer = 0
    batch_count = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        progress_bar = tqdm(loader, desc='Validation')
        for batch_data in progress_bar:
            try:
                if batch_data is None:
                    continue

                images, labels, label_lengths = batch_data
                images = images.to(device)
                labels = labels.to(device)
                label_lengths = label_lengths.to(device)

                # Forward pass
                outputs = model(images)
                batch_size = outputs.size(0)
                input_lengths = torch.full(
                    size=(batch_size,),
                    fill_value=outputs.size(1),
                    dtype=torch.long,
                    device=device
                )

                log_probs = outputs.log_softmax(2)
                loss = criterion(log_probs.permute(1, 0, 2), labels, input_lengths, label_lengths)

                # Get predictions
                predictions = log_probs.argmax(2)
                pred_texts = decode_predictions(predictions, label_lengths, dataset.dataset.idx_to_char)
                true_texts = decode_predictions(labels, label_lengths, dataset.dataset.idx_to_char)

                correct_chars = 0
                total_chars = 0
                batch_cer = 0

                for pred_text, true_text in zip(pred_texts, true_texts):
                    # Calculate accuracy
                    for p, t in zip(pred_text, true_text):
                        if p == t:
                            correct_chars += 1
                        total_chars += 1

                    # Calculate CER
                    max_len = max(len(pred_text), len(true_text))
                    if max_len > 0:
                        batch_cer += (abs(len(pred_text) - len(true_text)) +
                                      sum(p != t for p, t in
                                          zip(pred_text.ljust(max_len), true_text.ljust(max_len)))) / max_len

                accuracy = correct_chars / total_chars if total_chars > 0 else 0
                batch_cer = batch_cer / batch_size

                total_loss += loss.item()
                total_accuracy += accuracy
                total_cer += batch_cer
                batch_count += 1

                # Store sample predictions
                if len(all_predictions) < 10:
                    all_predictions.extend(pred_texts[:2])
                    all_targets.extend(true_texts[:2])

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{accuracy:.4f}',
                    'cer': f'{batch_cer:.4f}'
                })

            except Exception as e:
                logger.error(f"Error in validation batch: {str(e)}")
                logger.error(traceback.format_exc())
                continue

    # Calculate averages
    avg_loss = total_loss / max(1, batch_count)
    avg_accuracy = total_accuracy / max(1, batch_count)
    avg_cer = total_cer / max(1, batch_count)

    # Log to TensorBoard
    writer.add_scalar('Validation/Loss', avg_loss, epoch)
    writer.add_scalar('Validation/Accuracy', avg_accuracy, epoch)
    writer.add_scalar('Validation/CER', avg_cer, epoch)

    # Log sample predictions
    logger.info("\nValidation Samples:")
    for pred, target in zip(all_predictions, all_targets):
        logger.info(f"Pred: {pred}")
        logger.info(f"True: {target}")
        logger.info("-" * 50)

    return avg_loss, avg_accuracy, avg_cer


def save_checkpoint(epoch, model, optimizer, scaler, scheduler, metrics, checkpoint_dir, is_best=False):
    """Save model checkpoint with detailed metrics and training state."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'creation_time': "2025-01-26 19:54:20",
        'author': "CommanderBittu",
        'training_config': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'gradient_clip': GRADIENT_CLIP
        }
    }

    # Save epoch checkpoint
    epoch_filename = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, epoch_filename)

    # Save best model if needed
    if is_best:
        best_filename = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_filename)
        logger.info(f"Saved best model with metrics: {metrics}")


def setup_training():
    """Setup training environment and return necessary components."""
    try:
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

        # Initialize TensorBoard writer
        writer = SummaryWriter(f'runs/hindi_ocr_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

        # Initialize dataset
        logger.info("Initializing dataset...")
        torch.cuda.empty_cache()

        full_dataset = HindiOCRDataset(
            annotations_file=TRAIN_ANNOTATIONS,
            img_dir=TRAIN_IMG_DIR,
            transform=train_transform
        )

        # Split dataset
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

        logger.info(f"Dataset split - Train: {train_size}, Validation: {val_size}")

        # Create data loaders with multiprocessing settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=HindiCollator(full_dataset.char_to_idx),
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True,
            persistent_workers=True,
            multiprocessing_context='spawn'  # Use spawn context for Windows
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=HindiCollator(full_dataset.char_to_idx),
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True,
            multiprocessing_context='spawn'  # Use spawn context for Windows
        )

        # Initialize model
        model = FineTunedEasyOCR(num_chars=len(full_dataset.char_to_idx) + 1)
        model = model.to(device)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

        # Initialize optimizer and criterion
        optimizer = optim.AdamW(
            model.parameters(),
            lr=INITIAL_LR,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        scaler = amp.GradScaler(enabled=torch.cuda.is_available())

        # Initialize learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=LEARNING_RATE,
            epochs=NUM_EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,  # Warm up for 10% of training
            anneal_strategy='cos',
            div_factor=10.0,  # Initial LR = max_lr/10
            final_div_factor=100.0  # Final LR = max_lr/1000
        )

        return (device, writer, full_dataset, train_loader, val_loader,
                model, optimizer, criterion, scaler, scheduler)

    except Exception as e:
        logger.error(f"Error in setup: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def main():
    try:
        # Setup logging
        logger.info("Starting Hindi OCR training script...")
        logger.info(f"Script started by {SCRIPT_AUTHOR} at {SCRIPT_CREATION_TIME} UTC")

        # Initialize training components
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

        # Initialize dataset and model
        logger.info("Initializing dataset...")
        full_dataset = HindiOCRDataset(TRAIN_ANNOTATIONS, TRAIN_IMG_DIR, train_transform)

        # Split dataset
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

        logger.info(f"Dataset split - Train: {train_size}, Validation: {val_size}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=HindiCollator(full_dataset.label_processor),
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=HindiCollator(full_dataset.label_processor),
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )

        # Initialize model
        logger.info("Initializing Fine-Tuned EasyOCR model")
        model = FineTunedEasyOCR(num_chars=len(full_dataset.label_processor.chars) + 1)
        model = model.to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params}")

        # Initialize loss and optimizer
        criterion = nn.CTCLoss(zero_infinity=True)
        optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=LEARNING_RATE,
            epochs=NUM_EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.1
        )

        # Initialize TensorBoard
        writer = SummaryWriter(f'runs/hindi_ocr_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

        # Training loop
        logger.info("Starting training loop...")
        best_metrics = {'accuracy': 0, 'loss': float('inf'), 'cer': float('inf'), 'epoch': 0}
        start_time = datetime.now()

        try:
            for epoch in range(1, NUM_EPOCHS + 1):
                # Training
                train_loss, train_accuracy, train_cer = train_epoch(
                    model, train_loader, criterion, optimizer, device,
                    full_dataset.label_processor, epoch
                )

                # Validation
                val_loss, val_accuracy, val_cer = validate_epoch(
                    model, val_loader, criterion, device,
                    full_dataset.label_processor
                )

                # Update learning rate
                scheduler.step()

                # Log metrics
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Accuracy/train', train_accuracy, epoch)
                writer.add_scalar('Accuracy/val', val_accuracy, epoch)
                writer.add_scalar('CER/train', train_cer, epoch)
                writer.add_scalar('CER/val', val_cer, epoch)

                # Update best metrics
                if val_accuracy > best_metrics['accuracy']:
                    best_metrics = {
                        'accuracy': val_accuracy,
                        'loss': val_loss,
                        'cer': val_cer,
                        'epoch': epoch
                    }
                    # Save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_metrics': best_metrics,
                    }, 'models/best_model.pth')

                # Print epoch summary
                logger.info(f"Epoch {epoch}/{NUM_EPOCHS}")
                logger.info(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, CER: {train_cer:.4f}")
                logger.info(f"Val - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, CER: {val_cer:.4f}")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            logger.error(traceback.format_exc())

        finally:
            # Training summary
            training_time = datetime.now() - start_time
            logger.info(f"Training completed. Total time: {training_time}")
            logger.info(f"Best metrics achieved: {best_metrics}")
            writer.close()

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()