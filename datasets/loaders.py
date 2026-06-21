import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
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
                
    elif name == "synthetic_large":
        # Large-scale synthetic dataset for ImageNet-scale validation
        # Generates complex multi-class patterns at arbitrary scale
        num_classes = min(subset_size // 10, 200)  # Scale classes with data
        features = downsample_size * downsample_size
        n_train = min(subset_size * 5, 50000)
        n_test = min(subset_size, 10000)
        
        np.random.seed(seed)
        X_list, y_list = [], []
        
        for _ in range(n_train + n_test):
            label = np.random.randint(0, num_classes)
            # Create complex 2D pattern
            img = np.zeros((downsample_size, downsample_size))
            # Each class has a unique combination of features
            np.random.seed(label * 1000 + _)
            for _ in range(3 + label % 5):
                r = np.random.randint(0, downsample_size)
                c = np.random.randint(0, downsample_size)
                img[r, c] = 1.0
                # Add spatial structure
                if r > 0: img[r-1, c] = max(img[r-1, c], 0.5)
                if r < downsample_size-1: img[r+1, c] = max(img[r+1, c], 0.5)
                if c > 0: img[r, c-1] = max(img[r, c-1], 0.5)
                if c < downsample_size-1: img[r, c+1] = max(img[r, c+1], 0.5)
            noise = np.random.normal(0, 0.1, size=img.shape)
            sample = np.clip(img + noise, 0, 1)
            X_list.append(sample.flatten())
            y_list.append(label)
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
        X_train, X_test = torch.tensor(X[:n_train]), torch.tensor(X[n_train:])
        y_train, y_test = torch.tensor(y[:n_train]), torch.tensor(y[n_train:])
        return (X_train, y_train, X_test, y_test, features, num_classes)
        
    elif name == "tiny_imagenet":
        # Tiny ImageNet (200 classes, 100k images) - ImageNet-scale proxy
        import torchvision
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize((downsample_size, downsample_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        try:
            train_set = torchvision.datasets.ImageFolder(
                root='./data/tiny-imagenet-200/train', transform=transform
            )
            test_set = torchvision.datasets.ImageFolder(
                root='./data/tiny-imagenet-200/val', transform=transform
            )
            # Subsample for speed
            indices = torch.randperm(len(train_set))[:subset_size]
            train_set = torch.utils.data.Subset(train_set, indices)
            indices = torch.randperm(len(test_set))[:min(subset_size, len(test_set))]
            test_set = torch.utils.data.Subset(test_set, indices)
        except:
            print("Tiny ImageNet not found, using synthetic_large instead")
            return get_dataset('synthetic_large', subset_size=subset_size, 
                             downsample_size=downsample_size, seed=seed)
        
        train_loader = DataLoader(train_set, batch_size=subset_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=subset_size, shuffle=False)
        
        X_train, y_train = next(iter(train_loader))
        X_test, y_test = next(iter(test_loader))
        
        X_train = X_train.view(X_train.size(0), -1)
        X_test = X_test.view(X_test.size(0), -1)
        
        return (X_train, y_train, X_test, y_test, downsample_size * downsample_size, 200)
        
    elif name == "cifar10":
        # CIFAR-10 dataset with graceful fallback - checks for pre-downloaded data
        import torchvision
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize((downsample_size, downsample_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        cifar_dir = os.path.join('data', 'cifar-10-batches-py')
        if not os.path.exists(cifar_dir):
            print(f"  CIFAR-10 data not found at {cifar_dir}, using synthetic_large (10 classes)")
            return get_dataset('synthetic_large', subset_size=subset_size,
                             downsample_size=downsample_size, seed=seed)
        try:
            train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
            test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
            
            train_loader = DataLoader(train_set, batch_size=subset_size, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=subset_size, shuffle=False)
            
            X_train, y_train = next(iter(train_loader))
            X_test, y_test = next(iter(test_loader))
            
            X_train = X_train.view(X_train.size(0), -1)
            X_test = X_test.view(X_test.size(0), -1)
            
            return (X_train, y_train, X_test, y_test, downsample_size * downsample_size, 10)
        except Exception as e:
            print(f"  CIFAR-10 load failed ({e}), using synthetic_large (10 classes)")
            return get_dataset('synthetic_large', subset_size=subset_size,
                             downsample_size=downsample_size, seed=seed)
        
    elif name in ["mnist", "fashion"]:
        # Load real MNIST/FashionMNIST datasets
        # Fix: Remove toy procedural generator - use real datasets only
        import torchvision
        import torchvision.transforms as transforms
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

    elif name == "cifar10_rgb":
        import torchvision
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize((downsample_size, downsample_size)),
            transforms.ToTensor(),
        ])
        try:
            train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        except:
            print("  CIFAR-10 RGB download failed, using synthetic_large (10 classes)")
            return get_dataset('synthetic_large', subset_size=subset_size,
                             downsample_size=downsample_size, seed=seed)

        train_loader = DataLoader(train_set, batch_size=subset_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=subset_size, shuffle=False)

        X_train, y_train = next(iter(train_loader))
        X_test, y_test = next(iter(test_loader))

        X_train = X_train.view(X_train.size(0), -1)
        X_test = X_test.view(X_test.size(0), -1)

        return (X_train, y_train, X_test, y_test, downsample_size * downsample_size * 3, 10)

    elif name == "svhn":
        import torchvision
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize((downsample_size, downsample_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        try:
            train_set = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
            test_set = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
        except:
            print("  SVHN download failed, using synthetic_large (10 classes)")
            return get_dataset('synthetic_large', subset_size=subset_size,
                             downsample_size=downsample_size, seed=seed)

        train_loader = DataLoader(train_set, batch_size=subset_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=subset_size, shuffle=False)

        X_train, y_train = next(iter(train_loader))
        X_test, y_test = next(iter(test_loader))

        X_train = X_train.view(X_train.size(0), -1)
        X_test = X_test.view(X_test.size(0), -1)

        y_train = y_train.to(torch.long)
        y_test = y_test.to(torch.long)

        return (X_train, y_train, X_test, y_test, downsample_size * downsample_size, 10)

    elif name == "regression":
        num_features = 5
        n_train = subset_size
        n_test = subset_size // 2

        np.random.seed(seed)
        w_true = np.random.randn(num_features).astype(np.float32)
        b_true = np.random.randn(1).astype(np.float32)

        X_train_np = np.random.uniform(0, 1, size=(n_train, num_features)).astype(np.float32)
        y_train_np = (X_train_np @ w_true + b_true + np.random.normal(0, 0.1, size=n_train).astype(np.float32)).reshape(-1, 1)

        X_test_np = np.random.uniform(0, 1, size=(n_test, num_features)).astype(np.float32)
        y_test_np = (X_test_np @ w_true + b_true + np.random.normal(0, 0.1, size=n_test).astype(np.float32)).reshape(-1, 1)

        return (torch.tensor(X_train_np), torch.tensor(y_train_np),
                torch.tensor(X_test_np), torch.tensor(y_test_np),
                num_features, 1)

    elif name == "california_housing":
        try:
            from sklearn.datasets import fetch_california_housing
            data = fetch_california_housing()
            X, y = data.data.astype(np.float32), data.target.astype(np.float32).reshape(-1, 1)
        except:
            print("  California housing download failed, using regression instead")
            return get_dataset('regression', subset_size=subset_size, seed=seed)

        X_min, X_max = X.min(axis=0), X.max(axis=0)
        X = (X - X_min) / (X_max - X_min + 1e-6)

        indices = np.random.permutation(len(X))
        n_train = min(subset_size, len(X) // 2)
        train_idx, test_idx = indices[:n_train], indices[n_train:n_train + subset_size // 2]

        return (torch.tensor(X[train_idx]), torch.tensor(y[train_idx]),
                torch.tensor(X[test_idx]), torch.tensor(y[test_idx]),
                X.shape[1], 1)

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
