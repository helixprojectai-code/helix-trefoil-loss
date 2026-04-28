import numpy as np
import matplotlib.pyplot as plt

def omm_weierstrass(x, gamma=1/3, base=3, terms=100):
    y = np.zeros_like(x)
    for n in range(terms):
        y += (gamma ** n) * np.cos((base ** n) * np.pi * x)
    return y

x = np.linspace(-2, 2, 10000)
y = omm_weierstrass(x)

plt.figure(figsize=(14, 8))
plt.plot(x, y, 'b-', linewidth=1.5, label=r'$\Psi_{\Omega}(x) = \sum (1/3)^n \cos(3^n \pi x)$')

ax_inset = plt.axes([0.6, 0.6, 0.25, 0.25])
x_zoom = np.linspace(0.1, 0.15, 2000)
y_zoom = omm_weierstrass(x_zoom)
ax_inset.plot(x_zoom, y_zoom, 'r-', linewidth=1)
ax_inset.set_title("Microscopic Scale (0.1 to 0.15)", fontsize=10)
ax_inset.set_xticks([])
ax_inset.set_yticks([])

plt.suptitle("The Omm Postulate: Recursion All The Way Through Shape", fontsize=16, fontweight='bold')
plt.title("The Trefoil Phase-Locking Constraint expressed as a Continuous Fractal Wave", fontsize=12)
plt.xlabel("Space-Time ($x$)", fontsize=12)
plt.ylabel("Topological Amplitude", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# Fix legend by making sure it picks up the label
handles, labels = plt.gca().get_legend_handles_labels()
if handles:
    plt.legend(handles, labels, loc='lower left', fontsize=12)

textstr = r"Omm Intuition:\nIf shape is fractal, the $\gamma=1/3$ constraint\nmaps to the Weierstrass function.\nIt is continuous everywhere but differentiable nowhere.\nThe knot exists at all scales simultaneously."
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
plt.text(-1.9, 1.2, textstr, fontsize=11, bbox=props)

plt.savefig("/home/azureuser/lattice/research/omm_recursion/omm_fractal_wave.png", dpi=300)
print("✅ Fractal wave mapped. Plot saved to: /home/azureuser/lattice/research/omm_recursion/omm_fractal_wave.png")
