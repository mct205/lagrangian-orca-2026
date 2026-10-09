# Lagrangian statistics and coherent structures

Summer-school material on the Lagrangian analysis of ocean surface currents, from theory to HF-radar data.

## Notebooks

| Notebook | Content |
|---|---|
| [`00-lagrangian-theory.ipynb`](00-lagrangian-theory.ipynb) | **Theory.** Trajectories and the flow map; single-particle statistics (absolute dispersion, Lagrangian time scale, diffusivity); pair statistics (relative dispersion, kurtosis, FSLE as a function of scale); Lagrangian coherent structures (FTLE, FSLE maps, LAVD eddies); Okubo–Weiss for comparison; numerics. Equations only, no code. |
| [`01-lagrangian-bickley-jet.ipynb`](01-lagrangian-bickley-jet.ipynb) | **Idealised test with the Bickley jet.** Every diagnostic of the theory notebook applied to an analytic meandering jet, where the answer is known: chaotic stirring, a transport barrier, six coherent eddies. The jet parameters can be tuned in one cell. |
| [`02-lagrangian-hf-radar.ipynb`](02-lagrangian-hf-radar.ipynb) | **Test with HF-radar data.** The same functions applied to surface currents in the Gulf of Tonkin (January 2015, hourly, 6 km, gap-filled with the 2DVar method): tides, the need for continuous data, particles leaving the radar footprint, pair statistics, FTLE, FSLE and LAVD. |

Exercises are at the end of notebooks 01 and 02. Solutions for 01 are in [`solutions/`](solutions/).

## Setup

```bash
conda env create -f environment.yml
conda activate lagrangian-school
jupyter lab
```

`requirements.txt` lists the same packages for `pip`. Python 3.11 or later.

Run the notebooks from the repository root. Run times on a laptop: about 2 minutes for 01 (45 s with `FAST = True`) and 3 minutes for 02.

## Contents

| Path | What it is |
|---|---|
| `lagkit.py` | All the algorithms, shared by the notebooks: RK4 advection with exit times, an OceanParcels backend, dispersion statistics, FSLE, FTLE, LAVD, eddy extraction, Okubo–Weiss, data loading, gap filling, tidal filter, plotting helpers. |
| `tests/` | Unit tests for `lagkit.py`. Run `pytest`. |
| `data/hfr-got-2015-01.nc` | Gulf of Tonkin HF-radar currents, January 2015. |
| `data/coast_got_f.npz` | Coastline (GSHHS, full resolution) for the maps. |
| `animations/` | Animation of the Bickley jet (speed and vorticity) and the script that makes it. |
| `tools/` | Script that builds notebook 02 (`python tools/build_nb02.py`). |

## Licence

GPL-3.0, see [`LICENSE`](LICENSE).
