"""
OpenAnalogNN Interactive Demo
=============================
Streamlit app for exploring analog neural network simulation.
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="OpenAnalogNN Demo",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 OpenAnalogNN Interactive Demo")
st.markdown("Differentiable analog neural network simulation and calibration")

# ============================================================
# SIDEBAR: Configuration
# ============================================================
st.sidebar.header("Configuration")

dataset_name = st.sidebar.selectbox(
    "Dataset",
    ["xor", "iris", "mnist", "fashion", "cifar10_rgb", "svhn", "regression"],
    index=0,
)
subset_size = st.sidebar.slider("Subset size", 10, 500, 100, step=10)

st.sidebar.subheader("Model Architecture")
depth = st.sidebar.slider("Depth (layers)", 1, 6, 2)
width = st.sidebar.slider("Width (neurons/layer)", 8, 256, 32, step=8)
epochs = st.sidebar.slider("Training epochs", 5, 100, 20)

st.sidebar.subheader("Analog Non-Idealities")
noise_sigma = st.sidebar.slider("Weight noise σ", 0.0, 0.2, 0.05, 0.005)
mismatch = st.sidebar.slider("Resistor mismatch", 0.0, 0.1, 0.01, 0.005)
offset = st.sidebar.slider("Op-amp offset", 0.0, 0.01, 0.002, 0.001)
quan_bits = st.sidebar.slider("Quantization bits", 4, 16, 8)
vmax = st.sidebar.slider("Saturation V_max", 0.5, 5.0, 2.5, 0.5)

st.sidebar.subheader("Calibration")
cal_method = st.sidebar.selectbox(
    "Method",
    ["affine", "polynomial", "bayesian", "ensemble"],
    index=0,
)

# ============================================================
# MAIN CONTENT
# ============================================================

@st.cache_data
def load_data(name, size):
    from datasets.loaders import get_dataset
    return get_dataset(name, subset_size=size, seed=42)


with st.spinner("Loading dataset..."):
    try:
        X_train, y_train, X_test, y_test, nf, nc = load_data(dataset_name, subset_size)
        is_regression = (nc == 1)
        st.success(
            f"Loaded {dataset_name}: {nf} features, {nc} output(s), "
            f"{len(X_train)} train, {len(X_test)} test"
        )
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()

analog_config = {
    'noise_sigma': noise_sigma,
    'resistor_mismatch': mismatch,
    'opamp_offset': offset,
    'quantization_bits': quan_bits,
    'saturation_vmax': vmax,
    'enable_mismatch': True,
    'enable_noise': True,
    'enable_offset': True,
    'enable_quantization': True,
    'enable_saturation': True,
    'enable_drift': False,
    'enable_thermal': False,
    'enable_temperature': False,
}

if st.button("🚀 Train & Simulate"):
    from experiments.models import DigitalMLP, train_model
    from analog_layers.analog_linear import AnalogLinear

    progress_bar = st.progress(0)
    status_text = st.empty()

    hidden_dims = [width] * (depth - 1) if depth > 1 else []
    model = DigitalMLP(nf, hidden_dims, nc)

    status_text.text("Training digital baseline...")
    progress_bar.progress(0.2)

    if is_regression:
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader

        history = {
            'train_loss': [],
            'test_loss': [],
            'train_acc': [],
            'test_acc': [],
        }
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * bx.size(0)
            train_loss = epoch_loss / len(X_train)

            model.eval()
            with torch.no_grad():
                test_out = model(X_test)
                test_loss = criterion(test_out, y_test).item()

            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
    else:
        history = train_model(
            model, X_train, y_train, X_test, y_test,
            epochs=epochs, lr=0.01, batch_size=16, seed=42,
        )

    progress_bar.progress(0.4)
    status_text.text("Running analog simulation...")

    model.eval()
    with torch.no_grad():
        y_ideal = model(X_test)

    analog_model = DigitalMLP(nf, hidden_dims, nc, analog_config=analog_config)
    analog_model.load_state_dict(model.state_dict(), strict=False)
    analog_model.eval()

    with torch.no_grad():
        y_sim = analog_model(X_test)

    progress_bar.progress(0.6)
    status_text.text("Calibrating...")

    if cal_method == 'affine':
        from calibration.affine import AffineCalibrator
        cal = AffineCalibrator()
    elif cal_method == 'polynomial':
        from calibration.polynomial import PolynomialCalibrator
        cal = PolynomialCalibrator(degree=3)
    elif cal_method == 'bayesian':
        from calibration.bayesian import BayesianCalibrator
        cal = BayesianCalibrator()
    elif cal_method == 'ensemble':
        from calibration.ensemble import EnsembleCalibrator
        from calibration.affine import AffineCalibrator
        from calibration.polynomial import PolynomialCalibrator
        from calibration.learned import LearnedCalibrator
        cal = EnsembleCalibrator({
            'affine': AffineCalibrator(),
            'poly': PolynomialCalibrator(degree=2),
            'learned': LearnedCalibrator(epochs=50),
        }, strategy='average')

    try:
        cal.fit(y_sim, y_ideal)
        y_cal = cal.calibrate(y_sim)
    except Exception as e:
        st.warning(f"Calibration failed: {e}. Using uncalibrated output.")
        y_cal = y_sim

    progress_bar.progress(0.8)
    status_text.text("Computing metrics...")

    if is_regression:
        y_id_np = y_ideal.detach().cpu().numpy().flatten()
        y_sm_np = y_sim.detach().cpu().numpy().flatten()
        y_cl_np = y_cal.detach().cpu().numpy().flatten()
        y_tr_np = y_test.detach().cpu().numpy().flatten()

        mse_ideal = np.mean((y_id_np - y_tr_np) ** 2)
        mse_sim = np.mean((y_sm_np - y_tr_np) ** 2)
        mse_cal = np.mean((y_cl_np - y_tr_np) ** 2)
        rmse_pre = np.sqrt(np.mean((y_id_np - y_sm_np) ** 2))
        rmse_post = np.sqrt(np.mean((y_id_np - y_cl_np) ** 2))
        r2_ideal = 1 - np.sum((y_id_np - y_tr_np)**2) / np.sum((y_tr_np - y_tr_np.mean())**2)
        r2_sim = 1 - np.sum((y_sm_np - y_tr_np)**2) / np.sum((y_tr_np - y_tr_np.mean())**2)
        r2_cal = 1 - np.sum((y_cl_np - y_tr_np)**2) / np.sum((y_tr_np - y_tr_np.mean())**2)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Digital R²", f"{r2_ideal:.4f}")
        with col2:
            st.metric("Analog R²", f"{r2_sim:.4f}")
        with col3:
            st.metric("Calibrated R²", f"{r2_cal:.4f}")
        with col4:
            st.metric("RMSE (pre-cal)", f"{rmse_pre:.4f}")

        st.subheader("Calibration Parity (Regression)")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        vmin = min(y_id_np.min(), y_sm_np.min(), y_cl_np.min())
        vmax = max(y_id_np.max(), y_sm_np.max(), y_cl_np.max())

        ax1.scatter(y_id_np, y_sm_np, alpha=0.5, s=10, c='red', label='Pre-calibration')
        ax1.plot([vmin, vmax], [vmin, vmax], 'k--', alpha=0.5)
        ax1.set_xlabel('Ideal')
        ax1.set_ylabel('Simulated')
        ax1.set_title(f'Pre-Calibration (RMSE={rmse_pre:.4f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.scatter(y_id_np, y_cl_np, alpha=0.5, s=10, c='green', label='Post-calibration')
        ax2.plot([vmin, vmax], [vmin, vmax], 'k--', alpha=0.5)
        ax2.set_xlabel('Ideal')
        ax2.set_ylabel('Calibrated')
        ax2.set_title(f'Post-Calibration (RMSE={rmse_post:.4f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        st.pyplot(fig)
    else:
        from validation.metrics import compute_metrics
        sim_acc = compute_metrics(y_ideal, y_sim, y_cal, y_test)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Digital Accuracy", f"{sim_acc['accuracy_ideal']:.1%}")
        with col2:
            v = sim_acc['accuracy_sim']
            d = (v - sim_acc['accuracy_ideal']) * 100
            st.metric("Analog Accuracy", f"{v:.1%}", delta=f"{d:.1f}%")
        with col3:
            v = sim_acc['accuracy_calibrated']
            d = (v - sim_acc['accuracy_sim']) * 100
            st.metric("Calibrated Accuracy", f"{v:.1%}", delta=f"{d:.1f}%")
        with col4:
            st.metric("RMSE (pre-cal)", f"{sim_acc['rmse_pre_calibration']:.4f}")

        st.subheader("Calibration Parity")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        y_id_np = y_ideal.detach().cpu().numpy().flatten()
        y_sm_np = y_sim.detach().cpu().numpy().flatten()
        y_cl_np = y_cal.detach().cpu().numpy().flatten()

        ax1.scatter(y_id_np, y_sm_np, alpha=0.5, s=10, c='red', label='Pre-calibration')
        ax1.plot([y_id_np.min(), y_id_np.max()], [y_id_np.min(), y_id_np.max()], 'k--', alpha=0.5)
        ax1.set_xlabel('Ideal')
        ax1.set_ylabel('Simulated')
        ax1.set_title(f'Pre-Calibration (RMSE={sim_acc["rmse_pre_calibration"]:.4f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.scatter(y_id_np, y_cl_np, alpha=0.5, s=10, c='green', label='Post-calibration')
        ax2.plot([y_id_np.min(), y_id_np.max()], [y_id_np.min(), y_id_np.max()], 'k--', alpha=0.5)
        ax2.set_xlabel('Ideal')
        ax2.set_ylabel('Calibrated')
        ax2.set_title(f'Post-Calibration (RMSE={sim_acc["rmse_post_calibration"]:.4f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        st.pyplot(fig)

    # Training curves
    st.subheader("Training Curves")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['test_loss'], label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if not is_regression:
        ax2.plot(history['train_acc'], label='Train Accuracy')
        ax2.plot(history['test_acc'], label='Test Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.plot(np.log10(np.array(history['train_loss']) + 1e-10), label='Log10 Train Loss')
        ax2.plot(np.log10(np.array(history['test_loss']) + 1e-10), label='Log10 Test Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Log10 Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    st.pyplot(fig)

    # Scaling Law Insights (classification only)
    if not is_regression:
        st.subheader("📐 Scaling Law Insights")
        from nas.analog_nas import ScalingLawRobustnessScorer
        sl_scorer = ScalingLawRobustnessScorer(noise_sigma=noise_sigma)

        predicted_drop = sl_scorer.predict_drop(depth=depth, width=width)
        predicted_analog = sl_scorer.predict_accuracy(0.95, depth=depth, width=width)
        constraints = sl_scorer.get_architectural_constraints()
        rob_score = sl_scorer.robustness_score(depth, width, 0.95)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Accuracy Drop", f"{predicted_drop:.2%}")
        with col2:
            st.metric("Predicted Analog Accuracy", f"{predicted_analog:.1%}")
        with col3:
            st.metric("Robustness Score", f"{rob_score:.2%}")

        st.caption("drop = 0.130 × D^0.26 × W^0.18 × N^0.86 × exp(-0.35·log(D)·log(N))")

        constraints_df = {
            "Depth": list(constraints.keys()),
            "Max Noise σ": [
                f"{v:.5f}" if isinstance(v, float) and v < 1e3 else "unconstrained"
                for v in constraints.values()
            ],
        }
        st.table(constraints_df)
    else:
        st.info("📐 Scaling law insights are only available for classification tasks.")

    progress_bar.progress(1.0)
    status_text.text("Done!")

else:
    st.info("👈 Configure settings in the sidebar and click 'Train & Simulate'")

    st.subheader("📖 About OpenAnalogNN")
    st.markdown("""
**OpenAnalogNN** simulates analog neural network hardware non-idealities:

- **Resistor Mismatch**: $w_{eff} = w / (1 + \\delta)$, $\\delta \\sim \\mathcal{N}(0, \\sigma_R^2)$
- **Op-Amp Offset**: $y_{eff} = y + V_{os} \\cdot (1 + \\sum |w_{ij}|)$
- **Johnson-Nyquist Noise**: $v_{noise} = \\sqrt{4 k_B T R BW}$
- **TCR Drift**: $R(T) = R_0 (1 + \\alpha \\Delta T + \\beta \\Delta T^2)$
- **Quantization**: $y_q = \\text{round}(y \\cdot 2^b) / 2^b$
- **Saturation**: $y = \\text{clamp}(y, -V_{max}, V_{max})$

**Calibration Methods**: Affine, Polynomial, Bayesian GP, Ensemble (averaging/weighted/stacking)

**Scaling Law**: $\\text{drop} = 0.130 \\times D^{0.26} \\times W^{0.18} \\times N^{0.86} \\times \\exp(-0.35 \\cdot \\log(D) \\cdot \\log(N))$ $(R^2 = 0.9385)$
    """)

st.sidebar.markdown("---")
st.sidebar.info("🔬 OpenAnalogNN v0.2.0")
