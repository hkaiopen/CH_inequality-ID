# CH_inequality-ID

**Clauser-Horne Inequality Violation — Numerical Verification & Geometric Visualization**

This repository contains the numerical simulation code and geometric visualization scripts that accompany the paper:

> **"Why is the Clauser-Horne Inequality Violation 0.25? A Unified Geometric Explanation with First-Principles Numerical Verification Based on the Information Superconductor Model"**  
> *Kai Huang, Ziwei Huang (2026)*

The repository provides reproducible first-principles numerical verification of the linear projection mechanism — the core prediction of Information Dynamics — via 2D lattice FFT simulations of the generalized Ginzburg–Landau equation.

---

## 📁 Repository Structure

```
CH_inequality-ID/
├── CH Violation Predictions/
│   ├── Theoretical CH Violation Predictions.py      # Generates theoretical QM vs ID curves
│   └── Theoretical CH Violation Predictions_log.txt # Output log
├── Geometric Projections/
│   ├── Virtual Sphere with Geometric Projections.py # 3D visualization of virtual space curvature
│   └── virtual_sphere_geometry.png                  # Rendered figure
├── Numerical Verification/
│   ├── generate_table2_data.py                      # Main simulation: 2D FFT GL equation
│   ├── generate_table2_data_log.txt                 # Full run log (800 simulations)
│   ├── table2_raw_data.pkl                          # Raw simulation data (pickle)
│   └── table2_verification.png                      # Scatter plot: data vs α=1/α=2 curves
└── README.md
```

---

## 🧪 Numerical Verification (`Numerical Verification/`)

This is the **core** of the repository. It solves the generalized Ginzburg–Landau equation on a 2D periodic lattice using the FFT spectral method:

```
∂Ψ/∂t = D∇²Ψ − κ(|Ψ|² − ρ_target)Ψ + i K(x) Ψ
```

The coupling matrix `K(x)` is dynamically updated by local information density `ρ = |Ψ|²`, with **no linear or quadratic dependence prescribed**. The simulation extracts the CH violation `S_CH` as a function of concurrence `C`, then fits the data to:

```
S_CH(C) = (√(1 + C^α) − 1) / 2
```

- **α ≈ 1** → supports the **linear projection** (Information Dynamics)
- **α ≈ 2** → supports the **quadratic projection** (Standard Quantum Mechanics)

### Key Results

| Parameter Set (D, κ, ρ_target) | Mean α | α Std | Mean ΔAIC | ΔAIC < −2 |
|:---|:---:|:---:|:---:|:---:|
| (0.5, 1.0, 0.95) | 0.99 | 0.04 | −5.67 | 98% |
| (0.5, 1.0, 0.90) | 1.00 | 0.05 | −4.92 | 96% |
| (1.0, 0.5, 0.85) | 0.98 | 0.05 | −4.81 | 94% |
| (0.5, 1.0, 0.80) | 1.09 | 0.08 | −2.34 | 71% |
| (1.0, 2.0, 0.70) | 1.21 | 0.12 | +0.42 | 12% |

**Interpretation:** In the information superconducting phase (ρ_target ≳ 0.85), the linear projection (α ≈ 1) emerges naturally with overwhelming statistical support (ΔAIC < −2, 91% Monte Carlo support). The quadratic behavior of standard quantum mechanics only appears in the low-purity / strong-decoherence limit (ρ_target ≈ 0.70).

### Output Files

| File | Description |
|:---|:---|
| `table2_verification.png` | Scatter plot of simulated data vs theoretical α=1 and α=2 curves |
| `table2_raw_data.pkl` | Raw simulation data (Python pickle) for custom analysis |
| `generate_table2_data_log.txt` | Complete run log with progress and summary statistics |

---

## 📐 Geometric Projections (`Geometric Projections/`)

This script visualizes the **virtual space** as a 2D sphere (S²) with curvature K = 1/4, and illustrates the two geometric projections that produce the CH and I3322 bounds:

- **1D arc-length projection** → Tsirelson bound: (√2 − 1)/2 ≈ 0.2071
- **2D area projection** → I3322 upper bound: 0.25

The figure `virtual_sphere_geometry.png` is generated from this script.

---

## 📈 CH Violation Predictions (`CH Violation Predictions/`)

This script computes and plots the theoretical predictions for partially entangled states:

- **Standard Quantum Mechanics:** `S_QM(η) = (√(1 + sin² 2η) − 1) / 2`
- **Information Dynamics:** `S_ID(η) = (√(1 + sin 2η) − 1) / 2`

The two curves coincide at maximal entanglement (η = 45°) and diverge for η < 45°, with a maximum relative deviation exceeding 13% at η = 30°.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Required packages: `numpy`, `scipy`, `matplotlib`

### Installation

```bash
git clone https://github.com/hkaiopen/CH_inequality-ID.git
cd CH_inequality-ID
pip install numpy scipy matplotlib
```

### Running the Numerical Verification

```bash
cd "Numerical Verification"
python generate_table2_data.py
```

This will run the full simulation (800 runs, ~3–4 minutes on a typical laptop) and regenerate `table2_verification.png` and `table2_raw_data.pkl`.

### Running the Geometric Visualization

```bash
cd "Geometric Projections"
python "Virtual Sphere with Geometric Projections.py"
```

### Running the Theoretical Curve Generator

```bash
cd "CH Violation Predictions"
python "Theoretical CH Violation Predictions.py"
```

---

## 📖 Citation

If you use this code in your research, please cite the accompanying paper:

```bibtex
@article{Huang2026CH,
  title={Why is the Clauser-Horne Inequality Violation 0.25? A Unified Geometric Explanation with First-Principles Numerical Verification Based on the Information Superconductor Model},
  author={Huang, Kai and Huang, Ziwei},
  year={2026},
  note={Preprint}
}
```

---

## 📜 License

This project is licensed under the CC BY-NC.

---

你可以直接将上述内容保存为 `README.md` 并提交到仓库。如果你需要调整任何部分（例如添加更多运行细节、修改许可证类型等），随时告诉我。
