import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
import torchvision
import torchvision.transforms as transforms
import os

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_dataset(name: str, subset_size: int = 100, downsample_size: int = 8, seed: int = 42):
    """
    Returns a tuple of (X_train, y_train, X_test, y_test, num_features, num_classes)
    as PyTorch Tensors.
    """
    set_seed(seed)
    name = name.lower()
    
    if name == "xor":
        # Generate XOR dataset with slight noise
        X = np.random.uniform(-0.1, 1.1, size=(200, 2))
        y = (X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)
        y = y.astype(np.int64)
        
        # Split train/test
        X_train, X_test = X[:100], X[100:200]
        y_train, y_test = y[:100], y[100:200]
        return (torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long),
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long),
                2, 2)
                
    elif name == "iris":
        data = load_iris()
        X, y = data.data, data.target
        # Normalize inputs to [0, 1] range for analog mapping
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-6)
        
        indices = np.random.permutation(len(X))
        train_idx, test_idx = indices[:100], indices[100:]
        
        return (torch.tensor(X[train_idx], dtype=torch.float32),
                torch.tensor(y[train_idx], dtype=torch.long),
                torch.tensor(X[test_idx], dtype=torch.float32),
                torch.tensor(y[test_idx], dtype=torch.long),
                4, 3)
                
    elif name in ["mnist", "fashion"]:
        # Try downloading/loading using torchvision with offline fallback
        try:
            transform = transforms.Compose([
                transforms.Resize((downsample_size, downsample_size)),
                transforms.ToTensor()
            ])
            if name == "mnist":
                train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
                test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            else:
                train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
                test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
            
            # Extract subset
            train_loader = DataLoader(train_set, batch_size=subset_size, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=subset_size, shuffle=False)
            
            X_train, y_train = next(iter(train_loader))
            X_test, y_test = next(iter(test_loader))
            
            # Flatten images
            X_train = X_train.view(X_train.size(0), -1)
            X_test = X_test.view(X_test.size(0), -1)
            
            return (X_train, y_train, X_test, y_test, downsample_size * downsample_size, 10)
            
        except Exception as e:
            # Procedural digit/pattern generator fallback
            print(f"Torhvision dataset download failed/offline ({e}). Using robust procedural generator.")
            return generate_procedural_dataset(subset_size, downsample_size, seed)
            
    else:
        raise ValueError(f"Unknown dataset: {name}")

def generate_procedural_dataset(subset_size: int, downsample_size: int, seed: int):
    """
    Generates realistic, downsampled synthetic patterns for 10 classes
    (like digits 0-9) using combinations of lines and curves.
    """
    np.random.seed(seed)
    num_classes = 10
    features = downsample_size * downsample_size
    
    def create_digit_pattern(digit, size):
        img = np.zeros((size, size))
        mid = size // 2
        if digit == 0:
            img[1:size-1, 1] = 1.0
            img[1:size-1, size-2] = 1.0
            img[1, 1:size-1] = 1.0
            img[size-2, 1:size-1] = 1.0
        elif digit == 1:
            img[:, mid] = 1.0
        elif digit == 2:
            img[1, :] = 1.0
            img[mid, :] = 1.0
            img[-2, :] = 1.0
            img[1:mid, -2] = 1.0
            img[mid:size-1, 1] = 1.0
        elif digit == 3:
            img[1, :] = 1.0
            img[mid, :] = 1.0
            img[-2, :] = 1.0
            img[:, -2] = 1.0
        elif digit == 4:
            img[1:mid, 1] = 1.0
            img[:, -2] = 1.0
            img[mid, :] = 1.0
        elif digit == 5:
            img[1, :] = 1.0
            img[mid, :] = 1.0
            img[-2, :] = 1.0
            img[1:mid, 1] = 1.0
            img[mid:size-1, -2] = 1.0
        elif digit == 6:
            img[:, 1] = 1.0
            img[1, :] = 1.0
            img[mid, :] = 1.0
            img[-2, :] = 1.0
            img[mid:size-1, -2] = 1.0
        elif digit == 7:
            img[1, :] = 1.0
            img[:, -2] = 1.0
        elif digit == 8:
            img[:, 1] = 1.0
            img[:, -2] = 1.0
            img[1, :] = 1.0
            img[mid, :] = 1.0
            img[-2, :] = 1.0
        else: # 9
            img[1:mid, 1] = 1.0
            img[:, -2] = 1.0
            img[1, :] = 1.0
            img[mid, :] = 1.0
            img[-2, :] = 1.0
        return img

    X_list = []
    y_list = []
    
    # Generate train and test
    for _ in range(subset_size * 2):
        label = np.random.randint(0, num_classes)
        base = create_digit_pattern(label, downsample_size)
        # Add spatial noise and gaussian smoothing
        noise = np.random.normal(0, 0.15, size=base.shape)
        sample = np.clip(base + noise, 0, 1)
        X_list.append(sample.flatten())
        y_list.append(label)
        
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    
    X_train, X_test = X[:subset_size], X[subset_size:]
    y_train, y_test = y[:subset_size], y[subset_size:]
    
    return (torch.tensor(X_train), torch.tensor(y_train),
            torch.tensor(X_test), torch.tensor(y_test),
            features, num_classes)
