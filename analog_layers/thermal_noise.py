import torch


def apply_thermal_noise(activation: torch.Tensor, temperature_K: float = 300.0, resistance_ohm: float = 1e4, bandwidth_Hz: float = 1e6) -> torch.Tensor:
    """
    Applies Johnson-Nyquist thermal noise.

    v_noise_rms = sqrt(4 * k_B * T * R * BW)

    where:
        k_B = 1.380649e-23 J/K (Boltzmann constant)
        T = temperature in Kelvin
        R = resistance in Ohms
        BW = bandwidth in Hz

    At T=300K, R=10kOhm, BW=1MHz:
        v_noise = sqrt(4 * 1.38e-23 * 300 * 1e4 * 1e6) = 4.07e-4 V

    The noise is scaled relative to V_ref=1.0V, so relative noise approx 0.04%
    """
    k_B = 1.380649e-23
    v_noise_rms = torch.tensor(4.0 * k_B * temperature_K * resistance_ohm * bandwidth_Hz).sqrt()
    noise = torch.randn_like(activation) * v_noise_rms
    return activation + noise
