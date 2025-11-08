
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datareader import get_data_loaders, NEW_CLASS_NAMES
from model import ChestResNet
from utils import plot_training_history, visualize_random_val_predictions
from tqdm import tqdm

# --- Hyperparameters ---
EPOCHS = 10  # Reduced epochs for faster training
BATCH_SIZE = 32  # Increased batch size for faster processing
LEARNING_RATE = 0.001
NUM_WORKERS = 4  # Add worker threads for faster data loading
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner

def train(): 
    # 1. Load Data with num_workers for parallel loading
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    print("\n--- Checking Dataset ---")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    if len(train_loader) == 0 or len(val_loader) == 0:
        print("ERROR: Empty dataset or incorrect path!")
        return

    # 2. Initialize Model
    model = ChestResNet(in_channels=in_channels, num_classes=num_classes)
    model = model.to(DEVICE)
    print("\nModel ready:")
    print(model)

    # 3. Loss Function, Optimizer and Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Inisialisasi history training
    train_losses_history = []
    val_losses_history = []
    train_accs_history = []
    val_accs_history = []

    print("\n=== Memulai Training ===")
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print("="*50)

    # 4. Training Loop
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        # Progress bar untuk training
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = images.to(DEVICE)
            labels = labels.float().to(DEVICE)

            outputs = model(images).view(-1)
            labels = labels.view(-1)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # --- Validation Phase ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(DEVICE)
                labels = labels.float().to(DEVICE)

                outputs = model(images).view(-1)
                labels = labels.view(-1)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item()

                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        # Simpan history
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_accs_history.append(train_accuracy)
        val_accs_history.append(val_accuracy)

        # Print epoch results and update scheduler
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')

    print("\n=== Training Complete ===")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Final Validation Accuracy: {val_accs_history[-1]:.2f}%")
    print("="*50)
    
    # 5. Visualisasi hasil training
    plot_training_history(train_losses_history, val_losses_history, 
                         train_accs_history, val_accs_history)    # 6. Visualisasi prediksi acak dari validation set
    visualize_random_val_predictions(model, val_loader, num_classes, count=10, device=DEVICE)

if __name__ == '__main__':
    train()
