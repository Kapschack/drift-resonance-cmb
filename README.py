import numpy as np
import matplotlib.pyplot as plt

# Konstante Parameter
G = 4.302e-6  # kpc/M_sun (km/s)^2
lambda_ = 2e-7  # m/s^2 - schwache Geodrift-Kopplung
ell = 1.0       # kpc - Reichweite des Drift-Kerns
tau = 1.0e7     # Jahre - Relaxationszeit

# Exponentieller Gedächtnisterm
def memory_term(v_hist, t_hist, lambda_, tau):
    dt = t_hist[1] - t_hist[0]
    weights = np.exp(-(t_hist[-1] - t_hist) / tau)
    return lambda_ * np.sum(weights * v_hist) * dt

# Rotationsgeschwindigkeit mit Geodrift
def rotation_velocity(r, M_enc, lambda_, ell):
    v2_newton = G * M_enc / r
    v2_geodrift = lambda_ * ell * r
    return np.sqrt(v2_newton + v2_geodrift)

# Beispielhafte Massenverteilung (exponential disk)
def enclosed_mass(r, Sigma0=1000, Rd=3):
    return 2 * np.pi * Sigma0 * Rd**2 * (1 - np.exp(-r/Rd) * (1 + r/Rd))

# Plot
r_vals = np.linspace(0.1, 20, 200)
M_vals = enclosed_mass(r_vals)
v_vals = rotation_velocity(r_vals, M_vals, lambda_, ell)

plt.plot(r_vals, v_vals, label="Geodrift+Newton", color='blue')
plt.plot(r_vals, np.sqrt(G * M_vals / r_vals), label="Nur Newton", linestyle='--', color='gray')
plt.xlabel("Radius [kpc]")
plt.ylabel("v [km/s]")
plt.legend()
plt.grid(True)
plt.title("Rotationskurve mit Geodrift-Korrektur")
plt.show()
