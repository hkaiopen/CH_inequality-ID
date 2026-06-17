"""
Generate Table 1: Theoretical predictions for CH violation under partially entangled states

This script calculates S_QM and S_ID for a set of entanglement angles eta,
using the formulas:
    S_QM(eta) = (sqrt(1 + sin^2(2*eta)) - 1) / 2
    S_ID(eta) = (sqrt(1 + sin(2*eta)) - 1) / 2

The results are printed in a formatted table with columns:
    eta (deg), Concurrence C, QM prediction, ID prediction

Usage: python generate_table1.py
"""

import numpy as np

def S_QM(eta_deg):
    """Standard quantum mechanical prediction."""
    eta = np.deg2rad(eta_deg)
    sin2 = np.sin(2 * eta)
    return (np.sqrt(1 + sin2**2) - 1) / 2

def S_ID(eta_deg):
    """Information Dynamics prediction."""
    eta = np.deg2rad(eta_deg)
    sin2 = np.sin(2 * eta)
    return (np.sqrt(1 + sin2) - 1) / 2

# Angles to evaluate (in degrees)
eta_deg_values = [45, 40, 30, 20, 10]

# Print table header
print("Table 1: Theoretical predictions for CH violation under partially entangled states")
print("-" * 70)
print(f"{'η (deg)':<10} {'Concurrence C':<15} {'QM prediction':<15} {'ID prediction':<15}")
print("-" * 70)

# Compute and print each row
for eta in eta_deg_values:
    C = np.sin(2 * np.deg2rad(eta))
    s_qm = S_QM(eta)
    s_id = S_ID(eta)
    print(f"{eta:<10} {C:<15.3f} {s_qm:<15.4f} {s_id:<15.4f}")

print("-" * 70)
print("Note: At η=45° (maximal entanglement), both predictions coincide at 0.2071.")