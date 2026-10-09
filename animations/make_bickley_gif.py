"""Animated GIF of the Bickley jet: speed with velocity arrows (top), vorticity (bottom).

Run from the repository root:   python animations/make_bickley_gif.py
Edit the block below to change the jet (same names as in 01-lagrangian-bickley-jet.ipynb).
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, ".")
import lagkit as lk  # noqa: E402

# ---- flow (defaults of the course) ----
U0, L0, R0 = 62.66 * 86.4, 1770.0, 6371.0
EPS, C2, C3 = (0.075, 0.4, 0.3), 0.205, 0.461
# ---- animation ----
T_END = 50.0            # days
FRAME_DT = 0.5          # days between frames
FPS = 12
Y_MAX = 3000.0 * L0 / 1770.0
N_COLORS = 100          # colours per panel: a GIF holds 256 in total, so each colormap is discretised
OUT = "animations/bickley_jet_t0-50.gif"

bj = lk.BickleyJet(U0=U0, L0=L0, r0=R0, eps=EPS, c2=C2, c3=C3)
Lx = bj.Lx
x = np.linspace(0, Lx, 300, endpoint=False)
y = np.linspace(-Y_MAX, Y_MAX, 91)
X, Y = np.meshgrid(x, y)
times = np.arange(0, T_END + FRAME_DT / 2, FRAME_DT)
vmax = max(np.abs(bj.vorticity(X, Y, t)).max() for t in times[::10])
smax = max(np.hypot(*bj.velocity(X, Y, t)).max() for t in times[::10])
ext = (0, Lx, -Y_MAX, Y_MAX)
fig, (ax_s, ax_v) = plt.subplots(2, 1, figsize=(10, 6.4), dpi=80, sharex=True, constrained_layout=True)

# top: speed, with velocity arrows
u, v = bj.velocity(X, Y, 0.0)
im_s = ax_s.imshow(np.hypot(u, v), extent=ext, origin="lower", cmap=plt.get_cmap(lk.CMAPS["speed"], N_COLORS), vmin=0,
                   vmax=1.25 * smax, aspect="equal", interpolation="bilinear")
fig.colorbar(im_s, ax=ax_s, label="speed [km/day]", pad=0.02)
sk = (slice(None, None, 9), slice(None, None, 12))
qv = ax_s.quiver(X[sk], Y[sk], u[sk], v[sk], scale=16.6 * bj.U0, width=0.0016, color="tab:blue")
title_s = ax_s.set_title("")

# bottom: vorticity
im_v = ax_v.imshow(bj.vorticity(X, Y, 0.0), extent=ext, origin="lower", cmap=plt.get_cmap(lk.CMAPS["vorticity"], N_COLORS),
                   vmin=-vmax, vmax=vmax, aspect="equal", interpolation="bilinear")
fig.colorbar(im_v, ax=ax_v, label="vorticity [day$^{-1}$]", pad=0.02)
ax_v.set_title("Vorticity")
ax_v.set_xlabel("x [km] (periodic, frame moving with wave 3)")
for ax in (ax_s, ax_v):
    ax.set(ylabel="y [km]", xlim=(0, Lx), ylim=(-Y_MAX, Y_MAX))


def draw(k):
    t = times[k]
    u, v = bj.velocity(X, Y, t)
    im_s.set_data(np.hypot(u, v))
    qv.set_UVC(u[sk], v[sk])
    im_v.set_data(bj.vorticity(X, Y, t))
    title_s.set_text(f"Bickley jet, t = {t:4.1f} d: speed and velocity")
    return [im_s, qv, im_v, title_s]


FuncAnimation(fig, draw, frames=len(times), blit=False).save(OUT, writer=PillowWriter(fps=FPS))
print(f"{OUT}: {len(times)} frames, {len(times) / FPS:.0f} s at {FPS} fps")
