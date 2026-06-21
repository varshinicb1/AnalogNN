import numpy as np
import torch


class EnsembleCalibrator:
    """
    Ensemble calibration combining multiple calibrators.

    Three strategies:
    1. 'average': Simple average of all calibrator outputs
    2. 'weighted': Weighted average based on validation RMSE (lower RMSE = higher weight)
    3. 'stacking': Meta-model (linear regression) combining calibrator outputs

    Ensemble calibration is provably better than individual calibrators
    because it reduces both bias (through diverse hypothesis spaces) and
    variance (through averaging).

    For analog circuit calibration, this is especially powerful because:
    - Affine calibrator captures linear trends
    - Polynomial captures smooth non-linearities
    - Bayesian captures local structure with uncertainty
    - Different non-idealities favor different calibrator types
    """

    def __init__(self, calibrators: dict, strategy: str = 'average'):
        self.calibrators = calibrators
        self.strategy = strategy
        self._fitted = False
        self.weights = None
        self.stack_model = None

    def fit(self, y_spice: torch.Tensor, y_ideal: torch.Tensor):
        spice_np = y_spice.detach().cpu().numpy()
        ideal_np = y_ideal.detach().cpu().numpy()
        N = len(spice_np)

        split = N // 2
        indices = np.random.permutation(N)
        train_idx, val_idx = indices[:split], indices[split:]

        y_spice_train = torch.tensor(spice_np[train_idx])
        y_ideal_train = torch.tensor(ideal_np[train_idx])
        y_spice_val = torch.tensor(spice_np[val_idx])
        y_ideal_val = torch.tensor(ideal_np[val_idx])

        for name, cal in self.calibrators.items():
            cal.fit(y_spice_train, y_ideal_train)

        val_preds = {}
        for name, cal in self.calibrators.items():
            val_preds[name] = cal.calibrate(y_spice_val).detach().cpu().numpy()

        y_val_np = y_ideal_val.detach().cpu().numpy()

        if self.strategy == 'weighted':
            rmses = {}
            for name, pred in val_preds.items():
                rmse = np.sqrt(np.mean((pred - y_val_np) ** 2))
                rmses[name] = max(rmse, 1e-12)

            inv_rmse = np.array([1.0 / rmses[n] for n in self.calibrators.keys()])
            self.weights = inv_rmse / inv_rmse.sum()

        elif self.strategy == 'stacking':
            from sklearn.linear_model import LinearRegression

            n_val = len(val_idx)
            names = list(self.calibrators.keys())
            n_cal = len(names)

            num_classes = y_val_np.shape[1]
            X_stack = np.zeros((n_val, n_cal * num_classes))
            for j, name in enumerate(names):
                X_stack[:, j * num_classes:(j + 1) * num_classes] = val_preds[name]

            self.stack_model = LinearRegression()
            self.stack_model.fit(X_stack, y_val_np)

        self._fitted = True

    def calibrate(self, y_spice: torch.Tensor) -> torch.Tensor:
        if not self._fitted:
            raise ValueError("Calibrator has not been fitted!")

        all_preds = {}
        for name, cal in self.calibrators.items():
            all_preds[name] = cal.calibrate(y_spice)

        names = list(self.calibrators.keys())

        if self.strategy == 'average':
            result = sum(all_preds.values()) / len(self.calibrators)

        elif self.strategy == 'weighted':
            result = sum(self.weights[i] * all_preds[names[i]] for i in range(len(names)))

        elif self.strategy == 'stacking':
            y_spice_np = y_spice.detach().cpu().numpy()
            num_classes = y_spice_np.shape[1]
            n_val = len(y_spice_np)
            n_cal = len(names)

            X_stack = np.zeros((n_val, n_cal * num_classes))
            for j, name in enumerate(names):
                pred_np = all_preds[name].detach().cpu().numpy()
                X_stack[:, j * num_classes:(j + 1) * num_classes] = pred_np

            result_np = self.stack_model.predict(X_stack)
            result = torch.tensor(result_np, dtype=torch.float32)

        return result
