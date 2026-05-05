import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.special import ellipk, ellipkm1
from math import pi, sqrt

def Nmin1_RN(rho, z, M):
    return 1+M/sqrt(rho**2 + z**2)

def Nmin1_MP(rho, z, sigma, m):
    l2 = sqrt((rho + sigma)**2 + z**2)
    k2 = 4*sigma*rho/l2**2
    if k2 >= 0.95:
        K = ellipkm1(1-k2)
    else:
        K = ellipk(k2)
    return 1 + 2*m*K/(pi*l2)

n = 1/10
M = 1.0
m = 0.5*M
b = 20*M
sigma = b - M
eps = 0.955

rho = np.linspace(0.0001,30,1000)*M
l = np.linspace(2,4,6)*M

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

for j in range(len(l)):
    Veff = []
    urho_max = []
    rho_accessible = []

    for i in range(len(rho)):
        Nmin1_bh = Nmin1_RN(rho[i], n*sqrt(rho[i]), M)
        Nmin1_ring = Nmin1_MP(rho[i], n*sqrt(rho[i]), sigma, m)
        N = 1/(Nmin1_bh+Nmin1_ring-1)
        Veff.append(N**2+l[j]**2/rho[i]**2 *N**4)
        if eps**2-N**2-l[j]**2/rho[i]**2 *N**4 >= 0:
            urho_max.append(sqrt(eps**2-N**2-l[j]**2/rho[i]**2 *N**4))
            rho_accessible.append(rho[i])

    ax1.plot(rho,Veff, label=f"l = {l[j]}M")
    ax2.plot(rho_accessible, urho_max, label=f"l = {l[j]}M")

ax1.plot(rho,np.full_like(rho, eps**2), linestyle="--", color="black", label=fr"$\varepsilon^2$ = {eps**2}")
ax1.set_title(fr"$n$ = {n}, $z$ = $n$$\sqrt{{\rho}}$, $b$ = {b/M}$M$, $M$ = {M}, $m$ = {m/M}$M$")
ax1.legend()
ax1.set_xlabel(r'$\rho$ [M]')
ax1.set_ylabel(r'$\text{V}_{\text{eff}}$ [1]')
fig1.savefig("rn_mp_Veff", dpi=300)

ax2.set_title(fr"$\varepsilon$ = {eps}, $n$ = {n}, $z$ = $n$$\sqrt{{\rho}}$, $b$ = {b/M}$M$, $M$ = {M}, $m$ = {m/M}$M$")
ax2.legend()
ax2.set_xlabel(r'$\rho$ [M]')
ax2.set_ylabel(r'$u^\rho$ [1]')
fig2.savefig("rn_mp_accessible_region", dpi=300)

z = 0.2*M
rho = np.linspace(0.1,30,1000)*M

for j in range(len(l)):
    Veff = []
    urho_max = []
    rho_accessible = []

    for i in range(len(rho)):
        Nmin1_bh = Nmin1_RN(rho[i], z, M)
        Nmin1_ring = Nmin1_MP(rho[i], z, sigma, m)
        N = 1/(Nmin1_bh+Nmin1_ring-1)
        Veff.append(N**2+l[j]**2/rho[i]**2 *N**4)
        if eps**2-N**2-l[j]**2/rho[i]**2 *N**4 >= 0:
            urho_max.append(sqrt(eps**2-N**2-l[j]**2/rho[i]**2 *N**4))
            rho_accessible.append(rho[i])

    ax3.plot(rho,Veff, label=f"l = {l[j]}M")
    ax4.plot(rho_accessible, urho_max, label=f"l = {l[j]}M")

ax3.plot(rho,np.full_like(rho, eps**2), linestyle="--", color="black", label=fr"$\varepsilon^2$ = {eps**2}")
ax3.set_title(fr"$z$ = {z}$M$, $b$ = {b/M}$M$, $M$ = {M}, $m$ = {m/M}$M$")
ax3.legend()
ax3.set_xlabel(r'$\rho$ [M]')
ax3.set_ylabel(r'$\text{V}_{\text{eff}}$ [1]')
fig3.savefig("rn_mp_Veff_fix_z", dpi=300)

ax4.set_title(fr"$\varepsilon$ = {eps}, $n$ = {n}, $z$ = {z}, $b$ = {b/M}$M$, $M$ = {M}, $m$ = {m/M}$M$")
ax4.legend()
ax4.set_xlabel(r'$\rho$ [M]')
ax4.set_ylabel(r'$u^\rho$ [1]')
fig4.savefig("rn_mp_accessible_region_fix_z", dpi=300)