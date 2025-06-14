import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import expi
import astropy.units as u
import astropy.constants as const

# Astronomische Konstanten
G = const.G.to(u.kpc * u.km**2 / u.s**2 / u.M_sun).value  # Gravitationskonstante
c = const.c.to(u.km/u.s).value  # Lichtgeschwindigkeit

class GeodriftRotation:
    def __init__(self, mass=1e11, scale_radius=2.0, lambda_param=0.15, correlation_length=5.0):
        """Initialisiert die Galaxienrotationssimulation mit Geodrift-Feld
        
        Args:
            mass (float): Gesamtmasse in Sonnenmassen (10^11 M☉)
            scale_radius (float): Skalenradius in kpc
            lambda_param (float): Kopplungskonstante des Geodrift-Feldes
            correlation_length (float): Korrelationslänge in kpc
        """
        self.M = mass * 1e11  # Gesamtmasse in M☉
        self.R_d = scale_radius  # Skalenradius in kpc
        self.lmbda = lambda_param  # Geodrift-Kopplung
        self.ell = correlation_length  # Korrelationslänge in kpc
        
        # Baryonische Dichteprofile
        self.rho0 = self.M / (4 * np.pi * self.R_d**3)  # Zentraldichte
        
        # Numerische Parameter
        self.r_min = 0.1  # Minimale Galaxienradius (kpc)
        self.r_max = 30.0  # Maximale Galaxienradius (kpc)
    
    def baryonic_density(self, r):
        """Exponentielles Dichteprofil für baryonische Materie"""
        return self.rho0 * np.exp(-r / self.R_d)
    
    def newtonian_rotation(self, r):
        """Newton'sche Rotationskurve"""
        M_enc = self.M * (1 - (1 + r/self.R_d)*np.exp(-r/self.R_d))
        return np.sqrt(G * M_enc / r)
    
    def mond_rotation(self, r):
        """MOND-Rotationskurve mit Standard a0"""
        a0 = 1.2e-10  # MOND-Beschleunigungsparameter (m/s^2)
        a_newt = G * self.M * (1 - np.exp(-r/self.R_d)) / r**2
        return np.sqrt(r * np.sqrt(a_newt * a0))
    
    def geodrift_kernel(self, r, r_prime):
        """Nicht-lokaler Feldkern des Geodrift-Modells"""
        return np.exp(-np.abs(r - r_prime) / self.ell) / (4 * np.pi * self.ell**2 * np.abs(r - r_prime))
    
    def geodrift_potential(self, r):
        """Berechnet das Geodrift-Potential durch Integration"""
        # Numerische Integration des nicht-lokalen Terms
        r_prime = np.linspace(self.r_min, 10*self.R_d, 1000)
        dr_prime = r_prime[1] - r_prime[0]
        
        # Faltung des Kerns mit der baryonischen Dichte
        integral = np.sum(self.geodrift_kernel(r, r_prime) * 
                          self.baryonic_density(r_prime) * 
                          r_prime**2 * dr_prime * 4 * np.pi)
        
        # Gesamtpotential (Newton + Geodrift-Korrektur)
        phi_newt = -G * self.M * (1 - np.exp(-r/self.R_d)) / r
        phi_geo = phi_newt - self.lmbda * integral
        
        return phi_geo
    
    def geodrift_rotation(self, r):
        """Berechnet die Rotationskurve mit Geodrift-Feld"""
        # Numerische Differentiation des Potentials
        dr = 0.01 * r
        phi_plus = self.geodrift_potential(r + dr)
        phi_minus = self.geodrift_potential(r - dr)
        dphi_dr = (phi_plus - phi_minus) / (2 * dr)
        
        return np.sqrt(-r * dphi_dr)
    
    def simulate(self):
        """Führt die Simulation durch und gibt Ergebnisse zurück"""
        r = np.linspace(self.r_min, self.r_max, 100)
        
        results = {
            'radius': r,
            'newton': np.array([self.newtonian_rotation(ri) for ri in r]),
            'mond': np.array([self.mond_rotation(ri) for ri in r]),
            'geodrift': np.array([self.geodrift_rotation(ri) for ri in r])
        }
        
        return results
    
    def plot_results(self, results):
        """Visualisiert die Simulationsergebnisse"""
        plt.figure(figsize=(10, 6), dpi=100)
        
        # Rotationskurven
        plt.plot(results['radius'], results['newton'], 'b--', lw=2, label='Newton')
        plt.plot(results['radius'], results['mond'], 'g:', lw=2.5, label='MOND')
        plt.plot(results['radius'], results['geodrift'], 'r-', lw=3, label='Geodrift-Feld')
        
        # Beobachtete flache Rotationskurve
        plt.axhline(y=results['geodrift'][-1], color='k', linestyle='-', alpha=0.3, label='Beobachtete flache Kurve')
        
        # Einstellungen
        plt.title('Galaxienrotation im Geodrift-Feldmodell', fontsize=14)
        plt.xlabel('Radius (kpc)', fontsize=12)
        plt.ylabel('Rotationsgeschwindigkeit (km/s)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Parameterbox
        param_text = (f'M = {self.M/1e11:.1f} × 10$^{{{11}}}$ M$_\odot$\n'
                      f'R_d = {self.R_d:.1f} kpc\n'
                      f'λ = {self.lmbda:.2f}\n'
                      f'ℓ = {self.ell:.1f} kpc')
        plt.annotate(param_text, xy=(0.75, 0.15), xycoords='axes fraction', 
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('geodrift_rotation.png', dpi=150)
        plt.show()

# Hauptsimulation
if __name__ == "__main__":
    print("🚀 Simuliere Galaxienrotation mit Geodrift-Feld...")
    
    # Parameter der Milchstraßen-ähnlichen Galaxie
    simulation = GeodriftRotation(
        mass=1.5,       # 1.5 × 10^11 M☉
        scale_radius=3.0,  # 3.0 kpc
        lambda_param=0.12,  # Geodrift-Kopplungsstärke
        correlation_length=4.5  # Korrelationslänge in kpc
    )
    
    # Führe Simulation durch
    results = simulation.simulate()
    
    # Ergebnisse visualisieren
    simulation.plot_results(results)
    
    print("✅ Simulation abgeschlossen! Diagramm als 'geodrift_rotation.png' gespeichert.")
