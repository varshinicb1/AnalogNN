import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

class DigitalMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super(DigitalMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_model(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
                X_test: torch.Tensor, y_test: torch.Tensor,
                epochs: int = 50, lr: float = 0.01, batch_size: int = 16, seed: int = 42):
    """
    Trains the DigitalMLP model deterministically and tracks loss/accuracy curves.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)
            
        train_loss = epoch_loss / len(X_train)
        train_acc = correct / total
        
        # Validation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()
            _, test_pred = test_outputs.max(1)
            test_correct = test_pred.eq(y_test).sum().item()
            test_acc = test_correct / len(y_test)
            
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
    return history

def evaluate_model(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor):
    """
    Evaluates the model to return accuracy, confusion matrix, and raw predictions.
    """
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.softmax(logits, dim=1)
        _, preds = logits.max(1)
        
    accuracy = preds.eq(y_test).sum().item() / len(y_test)
    cm = confusion_matrix(y_test.numpy(), preds.numpy())
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'logits': logits,
        'predictions': preds,
        'probabilities': probs
    }

def plot_training_curves(history: dict, save_dir: str):
    """
    Saves a publication-grade training curve figure.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(epochs, history['test_loss'], label='Test Loss', color='#ff7f0e', linewidth=2, linestyle='--')
    ax1.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(frameon=True, facecolor='white')
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], label='Train Accuracy', color='#2ca02c', linewidth=2)
    ax2.plot(epochs, history['test_acc'], label='Test Accuracy', color='#d62728', linewidth=2, linestyle='--')
    ax2.set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(frameon=True, facecolor='white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
    plt.close()
