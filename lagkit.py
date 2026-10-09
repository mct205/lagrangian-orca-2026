"""lagkit -- shared Lagrangian diagnostics for the summer-school notebooks.

Everything works in a local Cartesian frame. Units are whatever the caller
uses consistently: the Bickley-jet notebook uses km and days, the HF-radar
practical km and hours.
In the docstrings ``[L]`` stands for the length unit and ``[T]`` for the time
unit; velocities are ``[L]/[T]``.

A *velocity function* is any callable ``vel(x, y, t) -> (u, v)`` where ``x`` and
``y`` are arrays of identical shape, ``t`` is a scalar (or an array broadcastable
to ``x``) and ``u``, ``v`` have the shape of ``x``. It must return NaN where the
field is undefined (outside the domain, on land, in a data gap): the integrator
then freezes the particle and records its exit time instead of extrapolating.

Two particle-tracking backends are provided and return the same things:

* :func:`advect`          -- vectorised numpy RK4 (the reference; shown in class)
* :func:`advect_parcels`  -- OceanParcels v4 on a gridded dataset
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "BickleyJet",
    "advect", "advect_parcels", "survival_mask",
    "grid_interpolator", "sample_to_dataset",
    "lonlat_to_km", "km_to_lonlat", "dataset_to_km",
    "absolute_dispersion", "lagrangian_autocorr", "integral_timescale",
    "pair_separation", "relative_dispersion", "fsle_statistic",
    "ftle_from_flowmap", "ftle_from_trajectories", "fsle_map", "lavd",
    "extract_eddies", "vorticity", "okubo_weiss", "stagnation_points", "liu_weisberg_skill",
    "load_surface_currents", "fill_gaps", "godin_filter", "ow_eddies", "advect_curve",
    "plot_field", "plot_eddies", "coastline", "plot_coastline", "CMAPS",
]

R_EARTH_KM = 6371.0

# Colormap convention shared by both notebooks (no rainbow/jet).
CMAPS = {
    "ftle_fwd": "Reds",
    "ftle_bwd": "Blues",
    "fsle": "Blues",
    "lavd": "Purples",
    "vorticity": "RdBu_r",
    "okubo_weiss": "PuOr_r",
    "speed": "Greys",
    "coverage": "Greens",
}


# ---------------------------------------------------------------------------
# Analytic test flow
# ---------------------------------------------------------------------------
class BickleyJet:
    """Meandering Bickley jet (Rypina et al. 2007), in the frame moving with c3.

    Stream function (u = -dpsi/dy, v = dpsi/dx)::

        psi = c3*y - U0*L0*tanh(y/L0)
              + U0*L0*sech^2(y/L0) * sum_n eps_n cos(k_n (x - (c_n - c3) t))

    Default units are km and days. The flow is periodic in x with period
    ``Lx = pi*r0`` and non-divergent.

    Parameters
    ----------
    U0 : float      jet speed [km/day] (62.66 m/s = 5413.8 km/day)
    L0 : float      jet half-width [km]
    r0 : float      Earth radius [km]
    eps : sequence of 3 floats, amplitudes of waves 1, 2, 3 [-]
    c2, c3 : float, phase speeds of waves 2 and 3 as fractions of U0. The
        speed of wave 1 follows from them (golden-mean rule of Rypina et al.),
        unless ``c1`` (also a fraction of U0) is given.

    Useful derived attributes: ``Lx`` (period), ``k`` (wavenumbers), ``c``
    (phase speeds [km/day]), ``sigma`` (phase speeds in the moving frame).
    """

    def __init__(self, U0=62.66 * 86.4, L0=1770.0, r0=6371.0,
                 eps=(0.075, 0.4, 0.3), c2=0.205, c3=0.461, c1=None):
        self.U0, self.L0, self.r0 = float(U0), float(L0), float(r0)
        self.eps = np.asarray(eps, dtype=float)
        self.Lx = np.pi * self.r0
        self.k = 2.0 * np.arange(1, 4) / self.r0
        c3 = c3 * self.U0
        c2 = c2 * self.U0
        if c1 is None:
            c1 = c3 + 0.5 * (np.sqrt(5.0) - 1.0) * (self.k[1] / self.k[0]) * (c2 - c3)
        else:
            c1 = c1 * self.U0
        self.c = np.array([c1, c2, c3])
        self.sigma = self.c - c3            # phase speeds in the moving frame

    def _parts(self, x, y, t):
        x, y, t = np.asarray(x, float), np.asarray(y, float), np.asarray(t, float)
        th = np.tanh(y / self.L0)
        sech2 = 1.0 - th ** 2
        S = SS = SK2 = 0.0
        for e, k, s in zip(self.eps, self.k, self.sigma):
            ph = k * (x - s * t)
            S = S + e * np.cos(ph)
            SS = SS + e * k * np.sin(ph)
            SK2 = SK2 + e * k ** 2 * np.cos(ph)
        return th, sech2, S, SS, SK2

    def psi(self, x, y, t):
        """Stream function [km^2/day]."""
        th, sech2, S, _, _ = self._parts(x, y, t)
        return self.c[2] * np.asarray(y, float) - self.U0 * self.L0 * th \
            + self.U0 * self.L0 * sech2 * S

    def velocity(self, x, y, t):
        """Velocity (u, v) [km/day]; ``t`` scalar or broadcastable to ``x``."""
        th, sech2, S, SS, _ = self._parts(x, y, t)
        u = -self.c[2] + self.U0 * sech2 + 2.0 * self.U0 * sech2 * th * S
        v = -self.U0 * self.L0 * sech2 * SS
        return u, v

    def vorticity(self, x, y, t):
        """Relative vorticity dv/dx - du/dy [1/day]."""
        th, sech2, S, _, SK2 = self._parts(x, y, t)
        return (-self.U0 * self.L0 * sech2 * SK2
                + 2.0 * self.U0 / self.L0 * sech2 * th
                - 2.0 * self.U0 / self.L0 * sech2 * (sech2 - 2.0 * th ** 2) * S)

    __call__ = velocity


# ---------------------------------------------------------------------------
# Particle advection (backend 1: numpy RK4)
# ---------------------------------------------------------------------------
def _rk4_step(x, y, t, h, vel):
    """One classical RK4 step of size ``h`` (signed). NaN in any stage -> NaN."""
    k1u, k1v = vel(x, y, t)
    k2u, k2v = vel(x + 0.5 * h * k1u, y + 0.5 * h * k1v, t + 0.5 * h)
    k3u, k3v = vel(x + 0.5 * h * k2u, y + 0.5 * h * k2v, t + 0.5 * h)
    k4u, k4v = vel(x + h * k3u, y + h * k3v, t + h)
    xn = x + h / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
    yn = y + h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return xn, yn


def _time_steps(T, dt):
    """Number of steps and signed step so that n*h == T exactly."""
    if dt <= 0:
        raise ValueError("dt must be positive; use the sign of T for direction")
    n = max(int(np.ceil(abs(T) / dt - 1e-9)), 1)
    return n, T / n


def _march(x0, y0, t0, T, dt, vel):
    """Generator behind all integrators.

    Yields ``(k, t, x, y, alive, tau)`` for k = 0..n, where ``alive`` flags
    particles still inside the valid field and ``tau`` is the elapsed time each
    particle has been integrated (>= 0, frozen at the exit time).
    """
    x = np.array(x0, dtype=float)
    y = np.array(y0, dtype=float)
    x, y = np.broadcast_arrays(x, y)
    x, y = x.copy(), y.copy()
    n, h = _time_steps(T, dt)
    alive = np.isfinite(x) & np.isfinite(y)
    tau = np.zeros(x.shape)
    yield 0, t0, x, y, alive, tau
    for k in range(n):
        t = t0 + k * h
        with np.errstate(invalid="ignore", over="ignore"):
            xn, yn = _rk4_step(x, y, t, h, vel)
        alive = alive & np.isfinite(xn) & np.isfinite(yn)
        x = np.where(alive, xn, x)          # frozen particles keep their position
        y = np.where(alive, yn, y)
        tau = np.where(alive, (k + 1) * abs(h), tau)
        yield k + 1, t0 + (k + 1) * h, x, y, alive, tau


def advect(x0, y0, t0, T, dt, vel, save_every=None, return_exit=False):
    """Advect particles with RK4, forward (T > 0) or backward (T < 0) in time.

    Parameters
    ----------
    x0, y0 : array_like, any (identical) shape, initial positions [L]
    t0 : float, release time [T]
    T : float, signed integration time [T]
    dt : float, positive time step [T] (adjusted slightly so n*dt == |T|)
    vel : callable ``vel(x, y, t) -> (u, v)`` in [L]/[T], NaN where undefined
    save_every : int or None
        If None return only the final positions. Otherwise store every
        ``save_every``-th step (plus the first and last).
    return_exit : bool
        Also return ``tau``, the elapsed time [T] each particle was integrated
        before leaving the valid field (``tau == |T|`` for survivors).

    Returns
    -------
    ``xf, yf`` (shape of x0)                         if ``save_every`` is None
    ``X, Y, t`` (shapes (nt, *x0.shape), (nt,))      otherwise
    with ``tau`` (shape of x0) appended if ``return_exit``.

    Particles whose velocity becomes NaN are frozen at their last valid
    position; they are never extrapolated.
    """
    n, _ = _time_steps(T, dt)
    X, Y, ts = [], [], []
    for k, t, x, y, alive, tau in _march(x0, y0, t0, T, dt, vel):
        if save_every is not None and (k % save_every == 0 or k == n):
            X.append(x), Y.append(y), ts.append(t)
    out = (x, y) if save_every is None else (np.array(X), np.array(Y), np.array(ts))
    return out + (tau,) if return_exit else out


def survival_mask(t, t0, tau):
    """Boolean mask (nt, *tau.shape): True while a particle is still valid.

    ``t`` are the saved times from :func:`advect`, ``tau`` its exit output.
    """
    el = np.abs(np.asarray(t, float) - t0)
    return el.reshape((-1,) + (1,) * np.ndim(tau)) <= np.asarray(tau)[None] + 1e-9


# ---------------------------------------------------------------------------
# Gridded fields
# ---------------------------------------------------------------------------
def lonlat_to_km(lon, lat, lon0, lat0):
    """Equirectangular projection about (lon0, lat0). Returns x, y in km."""
    x = R_EARTH_KM * np.cos(np.deg2rad(lat0)) * np.deg2rad(np.asarray(lon, float) - lon0)
    y = R_EARTH_KM * np.deg2rad(np.asarray(lat, float) - lat0)
    return x, y


def km_to_lonlat(x, y, lon0, lat0):
    """Inverse of :func:`lonlat_to_km`. Returns lon, lat in degrees."""
    lon = lon0 + np.rad2deg(np.asarray(x, float) / (R_EARTH_KM * np.cos(np.deg2rad(lat0))))
    lat = lat0 + np.rad2deg(np.asarray(y, float) / R_EARTH_KM)
    return lon, lat


def dataset_to_km(ds, lon="lon", lat="lat", time="time", lon0=None, lat0=None):
    """Put a regular (time, lat, lon) dataset on a local km grid.

    Returns a dataset with dims ``(time, y, x)``: ``x``, ``y`` in km about the
    domain centre (kept in ``attrs['lon0']``, ``attrs['lat0']``) and ``time`` in
    hours since the first sample (reference in ``attrs['t_ref']``) when the
    input time is datetime64. Data variables are untouched (still m/s).
    """
    lon0 = float(ds[lon].mean()) if lon0 is None else lon0
    lat0 = float(ds[lat].mean()) if lat0 is None else lat0
    x, _ = lonlat_to_km(ds[lon].values, lat0, lon0, lat0)
    _, y = lonlat_to_km(lon0, ds[lat].values, lon0, lat0)
    out = ds.rename({lon: "x", lat: "y", time: "time"}).transpose("time", "y", "x", ...)
    # coordinates stored with few decimals are not exactly uniform: snap them (shifts of metres)
    for c in (x, y):
        lin = np.linspace(c[0], c[-1], len(c))
        if len(c) > 2 and np.abs(c - lin).max() < 0.01 * abs(lin[1] - lin[0]):
            c[:] = lin
    out = out.assign_coords(x=x, y=y)
    out.attrs.update(lon0=lon0, lat0=lat0)
    tv = out["time"].values
    if np.issubdtype(tv.dtype, np.datetime64):
        out.attrs["t_ref"] = str(tv[0])
        out = out.assign_coords(time=(tv - tv[0]) / np.timedelta64(1, "h"))
    return out.sortby("y").sortby("x")


def grid_interpolator(ds_km, var_u, var_v=None, scale=1.0, x="x", y="y", t="time"):
    """Trilinear (t, y, x) interpolator of gridded fields.

    Parameters
    ----------
    ds_km : xarray.Dataset with 1-D, uniformly spaced coords ``x``, ``y`` [L]
        and an increasing numeric ``time`` coord [T] (need not be uniform).
    var_u, var_v : str
        Variable names. With ``var_v=None`` a scalar interpolator
        ``f(x, y, t) -> value`` is returned (e.g. for vorticity).
    scale : float
        Multiplies the data, e.g. 3.6 for m/s -> km/h, 86.4 for m/s -> km/day.

    Returns
    -------
    ``vel(x, y, t) -> (u, v)`` (or scalar function). The result is NaN outside
    the grid, outside the time range and wherever any of the 8 surrounding
    nodes is NaN, so gaps and land kill particles instead of being smeared.
    """
    names = [var_u] if var_v is None else [var_u, var_v]
    xg = np.asarray(ds_km[x].values, float)
    yg = np.asarray(ds_km[y].values, float)
    tg = np.asarray(ds_km[t].values, float)
    fields = [np.asarray(ds_km[n].transpose(t, y, x).values, float) * scale for n in names]
    nt, ny, nx = fields[0].shape
    dx, dy = (xg[-1] - xg[0]) / (nx - 1), (yg[-1] - yg[0]) / (ny - 1)
    if not (np.allclose(np.diff(xg), dx) and np.allclose(np.diff(yg), dy)):
        raise ValueError("x and y must be uniformly spaced")
    if nt > 1 and not np.all(np.diff(tg) > 0):
        raise ValueError("time must be strictly increasing")
    tol = 1e-9
    kgrid = np.arange(nt, dtype=float)

    def interp(xq, yq, tq):
        xq, yq, tq = np.broadcast_arrays(np.asarray(xq, float), np.asarray(yq, float),
                                         np.asarray(tq, float))
        fx = (xq - xg[0]) / dx
        fy = (yq - yg[0]) / dy
        tspan = max(tg[-1] - tg[0], 1.0)
        tc = np.where(np.abs(tq - tg[0]) < tol * tspan, tg[0], tq)   # round-off guard
        tc = np.where(np.abs(tc - tg[-1]) < tol * tspan, tg[-1], tc)
        ft = np.interp(tc, tg, kgrid, left=np.nan, right=np.nan) if nt > 1 \
            else np.where(np.isfinite(tc), 0.0, np.nan)
        inside = (np.isfinite(ft) & (fx >= -tol) & (fx <= nx - 1 + tol)
                  & (fy >= -tol) & (fy <= ny - 1 + tol))
        fx = np.clip(np.where(inside, fx, 0.0), 0, nx - 1)
        fy = np.clip(np.where(inside, fy, 0.0), 0, ny - 1)
        ft = np.where(inside, ft, 0.0)
        i = np.minimum(fx.astype(int), nx - 2)
        j = np.minimum(fy.astype(int), ny - 2)
        k = np.minimum(ft.astype(int), max(nt - 2, 0))
        k1 = np.minimum(k + 1, nt - 1)
        wx, wy, wt = fx - i, fy - j, ft - k
        out = []
        for F in fields:
            lo = ((F[k, j, i] * (1 - wx) + F[k, j, i + 1] * wx) * (1 - wy)
                  + (F[k, j + 1, i] * (1 - wx) + F[k, j + 1, i + 1] * wx) * wy)
            hi = ((F[k1, j, i] * (1 - wx) + F[k1, j, i + 1] * wx) * (1 - wy)
                  + (F[k1, j + 1, i] * (1 - wx) + F[k1, j + 1, i + 1] * wx) * wy)
            out.append(np.where(inside, lo * (1 - wt) + hi * wt, np.nan))
        return out[0] if var_v is None else tuple(out)

    interp.x, interp.y, interp.t = xg, yg, tg
    return interp


def sample_to_dataset(vel, x, y, t):
    """Sample a velocity function on a regular grid -> Dataset (time, y, x).

    Handy to feed an analytic flow to :func:`grid_interpolator` or
    :func:`advect_parcels`. Variables ``u``, ``v`` keep the units of ``vel``.
    """
    import xarray as xr
    x, y, t = (np.asarray(a, float) for a in (x, y, t))
    Xg, Yg = np.meshgrid(x, y)
    uv = [vel(Xg, Yg, ti) for ti in t]
    return xr.Dataset(
        {"u": (("time", "y", "x"), np.array([a[0] for a in uv])),
         "v": (("time", "y", "x"), np.array([a[1] for a in uv]))},
        coords={"time": t, "y": y, "x": x})


# ---------------------------------------------------------------------------
# Single-particle statistics
# ---------------------------------------------------------------------------
def absolute_dispersion(X, Y, valid=None):
    """Absolute dispersion about the ensemble-mean displacement.

    Parameters
    ----------
    X, Y : arrays (nt, npart) of positions [L]
    valid : bool array (nt, npart), optional. Only surviving particles enter the
        averages (see :func:`survival_mask`).

    Returns
    -------
    Ax2, Ay2 : arrays (nt,) [L^2], zonal and meridional dispersion
    n : array (nt,), number of particles in the average
    """
    X = np.asarray(X, float).reshape(len(X), -1)
    Y = np.asarray(Y, float).reshape(len(Y), -1)
    dX, dY = X - X[0], Y - Y[0]
    if valid is not None:
        m = np.asarray(valid, bool).reshape(len(X), -1)
        dX, dY = np.where(m, dX, np.nan), np.where(m, dY, np.nan)
    n = np.isfinite(dX).sum(axis=1)
    with np.errstate(invalid="ignore"), _quiet():
        Ax2 = np.nanmean((dX - np.nanmean(dX, axis=1, keepdims=True)) ** 2, axis=1)
        Ay2 = np.nanmean((dY - np.nanmean(dY, axis=1, keepdims=True)) ** 2, axis=1)
    return Ax2, Ay2, n


def lagrangian_autocorr(q, max_lag, valid=None):
    """Normalised Lagrangian autocorrelation of a velocity component.

    Parameters
    ----------
    q : array (nt, npart), Lagrangian velocity sampled at constant interval
    max_lag : int, largest lag in samples (< nt)
    valid : bool array like ``q``, optional

    Returns
    -------
    R : array (max_lag + 1,), with R[0] = 1. The ensemble-and-time mean is
    removed first (stationarity and homogeneity are assumed).
    """
    q = np.asarray(q, float).reshape(len(q), -1)
    if valid is not None:
        q = np.where(np.asarray(valid, bool).reshape(q.shape), q, np.nan)
    qp = q - np.nanmean(q)
    R = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        R[lag] = np.nanmean(qp[: len(qp) - lag] * qp[lag:])
    return R / R[0]


def integral_timescale(R, dtau):
    """Integral time scale: trapezoidal integral of R up to its first zero crossing.

    Returns ``(T_L, i_zero)``; ``i_zero`` is the index of the first R <= 0
    (or ``len(R)`` if R stays positive, in which case T_L is a lower bound).
    """
    R = np.asarray(R, float)
    neg = np.nonzero(R <= 0)[0]
    if len(neg) == 0:
        return np.trapezoid(R, dx=dtau), len(R)
    i = neg[0]
    frac = R[i - 1] / (R[i - 1] - R[i])               # linear zero crossing
    return np.trapezoid(R[:i], dx=dtau) + 0.5 * R[i - 1] * frac * dtau, i


# ---------------------------------------------------------------------------
# Two-particle statistics
# ---------------------------------------------------------------------------
def pair_separation(XA, YA, XB, YB, periodic_x=None):
    """Separation distance of particle pairs.

    Parameters
    ----------
    XA, YA, XB, YB : arrays (nt, npairs) [L]
    periodic_x : float, optional. Period in x; the minimum-image convention is
        then applied to the x separation.

    Returns
    -------
    D : array (nt, npairs) [L]
    """
    dX = np.asarray(XA, float) - np.asarray(XB, float)
    dY = np.asarray(YA, float) - np.asarray(YB, float)
    if periodic_x is not None:
        dX = (dX + 0.5 * periodic_x) % periodic_x - 0.5 * periodic_x
    return np.hypot(dX, dY)


def relative_dispersion(D, t, valid=None):
    """Relative dispersion, relative diffusivity and kurtosis.

    Parameters
    ----------
    D : array (nt, npairs) of separations [L];  t : array (nt,) [T]
    valid : bool array like ``D``, optional (both particles of the pair alive)

    Returns
    -------
    D2 : <D^2> [L^2];  Krel : 0.5 d<D^2>/dt [L^2/T];  kurt : <D^4>/<D^2>^2;
    n : pairs in the average. All of shape (nt,).
    """
    D = np.asarray(D, float)
    if valid is not None:
        D = np.where(valid, D, np.nan)
    n = np.isfinite(D).sum(axis=1)
    with _quiet():
        D2 = np.nanmean(D ** 2, axis=1)
        D4 = np.nanmean(D ** 4, axis=1)
    Krel = 0.5 * np.gradient(D2, np.asarray(t, float))
    return D2, Krel, D4 / D2 ** 2, n


def fsle_statistic(D, t, delta0, r=np.sqrt(2.0), n_max=None):
    """Finite-size Lyapunov exponent as a pair statistic (Aurell et al. 1997).

    Thresholds ``delta_n = delta0 * r**n``. For each pair, ``tau_n`` is the time
    between the first crossings of ``delta_n`` and ``delta_{n+1}`` (crossing
    times are linearly interpolated between samples). Then
    ``lambda(delta_n) = ln(r) / <tau_n>``, averaged over the pairs that reach
    ``delta_{n+1}`` -- report the pair count, it is a survivorship-biased sample.

    Pairs may be released at different separations (for example delta0, delta0*r,
    delta0*r**2, ...) to populate the large scales: a pair contributes to a
    threshold only if it started at or below it.

    Parameters
    ----------
    D : array (nt, npairs) [L]; NaN after a pair is lost
    t : array (nt,) [T];  delta0 : float [L], smallest threshold;  r : float > 1

    Returns
    -------
    delta : (n,) [L];  lam : (n,) [1/T];  npairs : (n,) int;  tau_mean : (n,) [T]
    """
    D = np.asarray(D, float)
    t = np.abs(np.asarray(t, float) - t[0])
    with np.errstate(invalid="ignore"):
        dmax = np.nanmax(D)
    if n_max is None:
        n_max = max(int(np.floor(np.log(dmax / delta0) / np.log(r))), 1)
    deltas = delta0 * r ** np.arange(n_max + 1)
    cols = np.arange(D.shape[1])
    tc = np.full((n_max + 1, D.shape[1]), np.nan)       # first-crossing times
    for n, d in enumerate(deltas):
        hit = D >= d
        i = hit.argmax(axis=0)
        ok = hit.any(axis=0) & (D[0] <= d * (1 + 1e-6))     # a pair released above d never "crossed" it
        i0 = np.maximum(i - 1, 0)
        Da, Db = D[i0, cols], D[i, cols]
        with np.errstate(invalid="ignore", divide="ignore"):
            w = np.where((i > 0) & (Db > Da), (d - Da) / (Db - Da), 1.0)
        w = np.where(np.isfinite(w), np.clip(w, 0.0, 1.0), 1.0)
        tc[n] = np.where(ok, t[i0] + w * (t[i] - t[i0]), np.nan)
    tau = tc[1:] - tc[:-1]
    npairs = np.isfinite(tau).sum(axis=1)
    with _quiet():
        tau_mean = np.nanmean(tau, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.log(r) / tau_mean
    return deltas[:-1], lam, npairs, tau_mean


# ---------------------------------------------------------------------------
# FTLE / FSLE maps
# ---------------------------------------------------------------------------
def _ddx(A, dx, periodic_x=None):
    """Centred x-derivative on a (ny, nx) grid; one-sided at non-periodic edges."""
    if periodic_x is None:
        return np.gradient(A, dx, axis=1)
    return (np.roll(A, -1, axis=1) - np.roll(A, 1, axis=1)) / (2.0 * dx)


def _ftle_from_gradients(Fxx, Fxy, Fyx, Fyy, T):
    """FTLE from the flow-map gradient; closed-form largest eigenvalue of C."""
    C11 = Fxx ** 2 + Fyx ** 2
    C12 = Fxx * Fxy + Fyx * Fyy
    C22 = Fxy ** 2 + Fyy ** 2
    lam2 = 0.5 * (C11 + C22) + np.sqrt(0.25 * (C11 - C22) ** 2 + C12 ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(lam2) / (2.0 * np.abs(T))


def ftle_from_flowmap(XF, YF, dx, dy, T, periodic_x=None, tau=None, min_frac=0.5):
    """Finite-time Lyapunov exponent from final positions of a grid of particles.

    ``sigma = ln(lambda_max(C)) / (2|T|)`` with ``C = (grad F)^T grad F`` and the
    gradient by centred differences on the *initial* grid.

    Parameters
    ----------
    XF, YF : arrays (ny, nx), final positions [L] of particles released on a
        regular grid with spacing ``dx``, ``dy`` [L]
    T : float, integration time [T] (sign ignored)
    periodic_x : float, optional. Period in x: the grid must cover exactly one
        period without repeating the end point and ``XF`` must be unwrapped.
    tau : array (ny, nx), optional. Per-particle integration time from
        ``advect(..., return_exit=True)``. The time used at each point is then
        the *minimum* of ``tau`` over its finite-difference stencil and points
        with less than ``min_frac * |T|`` are masked (NaN). This is approximate
        because stencil members stopped at different times; use
        :func:`ftle_from_trajectories` for the consistent version.

    Returns
    -------
    ftle : (ny, nx) [1/T]     (and ``tau_eff`` (ny, nx) if ``tau`` is given)
    """
    XF, YF = np.asarray(XF, float), np.asarray(YF, float)
    if periodic_x is not None:
        # unwrap the seam: neighbours across it are one period apart
        XL = np.roll(XF, 1, axis=1);  XL[:, 0] -= periodic_x
        XR = np.roll(XF, -1, axis=1); XR[:, -1] += periodic_x
        Fxx = (XR - XL) / (2.0 * dx)
    else:
        Fxx = np.gradient(XF, dx, axis=1)
    Fyx = _ddx(YF, dx, periodic_x)
    Fxy = np.gradient(XF, dy, axis=0)
    Fyy = np.gradient(YF, dy, axis=0)
    if tau is None:
        return _ftle_from_gradients(Fxx, Fxy, Fyx, Fyy, T)
    tau_eff = _stencil_min(np.asarray(tau, float), periodic_x is not None)
    with np.errstate(divide="ignore", invalid="ignore"):
        ftle = _ftle_from_gradients(Fxx, Fxy, Fyx, Fyy, tau_eff)
    ftle[~(tau_eff >= min_frac * abs(T))] = np.nan
    return ftle, tau_eff


def _stencil_min(tau, periodic):
    """Minimum of ``tau`` over the 5-point stencil."""
    p = np.pad(tau, 1, mode="edge")
    if periodic:
        p[1:-1, 0], p[1:-1, -1] = tau[:, -1], tau[:, 0]
    return np.minimum.reduce([p[1:-1, 1:-1], p[1:-1, :-2], p[1:-1, 2:],
                              p[:-2, 1:-1], p[2:, 1:-1]])


def ftle_from_trajectories(X, Y, t, tau, dx, dy, min_frac=0.5):
    """FTLE for open domains, consistent with particles leaving the field.

    At each grid point the flow map is evaluated at the last saved time at
    which *all* members of its finite-difference stencil were still valid, so
    every gradient uses positions taken at one common time.

    Parameters
    ----------
    X, Y : arrays (nt, ny, nx), saved trajectories from ``advect(save_every=..)``
    t : array (nt,), saved times [T];  tau : array (ny, nx), exit output [T]
    dx, dy : initial grid spacing [L]
    min_frac : mask points integrated for less than ``min_frac * |T|``

    Returns
    -------
    ftle : (ny, nx) [1/T];  tau_eff : (ny, nx) integration time actually used [T]
    """
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    el = np.abs(np.asarray(t, float) - t[0])
    tmin = _stencil_min(np.asarray(tau, float), False)
    k = np.clip(np.searchsorted(el, tmin + 1e-9, side="right") - 1, 0, len(el) - 1)
    tau_eff = el[k]

    def at(A, sj, si):
        """A at each point's own time index k, shifted by (sj, si) with edge clamp."""
        ny, nx = k.shape
        jj = np.clip(np.arange(ny)[:, None] + sj, 0, ny - 1)
        ii = np.clip(np.arange(nx)[None, :] + si, 0, nx - 1)
        return A[k, jj, ii]

    wx = np.full(k.shape, 2.0 * dx); wx[:, [0, -1]] = dx      # one-sided at edges
    wy = np.full(k.shape, 2.0 * dy); wy[[0, -1], :] = dy
    Fxx = (at(X, 0, 1) - at(X, 0, -1)) / wx
    Fyx = (at(Y, 0, 1) - at(Y, 0, -1)) / wx
    Fxy = (at(X, 1, 0) - at(X, -1, 0)) / wy
    Fyy = (at(Y, 1, 0) - at(Y, -1, 0)) / wy
    with np.errstate(divide="ignore", invalid="ignore"):
        ftle = _ftle_from_gradients(Fxx, Fxy, Fyx, Fyy, tau_eff)
    ftle[~(tau_eff >= min_frac * el[-1]) | (tau_eff <= 0)] = np.nan
    return ftle, tau_eff


def fsle_map(x0g, y0g, t0, T, dt, delta0, deltaf, vel):
    """Finite-size Lyapunov exponent map (d'Ovidio et al. 2004).

    Each grid point gets 4 neighbours at distance ``delta0`` (east, west, north,
    south). ``tau`` is the first time the largest centre-neighbour separation
    reaches ``deltaf`` (linearly interpolated within the step) and
    ``lambda = ln(deltaf/delta0) / tau``.

    Parameters
    ----------
    x0g, y0g : arrays, any identical shape, release positions [L]
    t0, T, dt : release time, signed maximum integration time, step [T]
    delta0, deltaf : initial and final separations [L]
    vel : velocity function

    Returns
    -------
    lam : [1/T]; 0 where ``deltaf`` was never reached within |T|, NaN where a
        member left the valid field first
    tau : [T]; NaN unless reached
    status : int array, 0 = never separated, 1 = reached, 2 = exited first
    """
    x0g, y0g = np.broadcast_arrays(np.asarray(x0g, float), np.asarray(y0g, float))
    ox = np.array([0.0, delta0, -delta0, 0.0, 0.0]).reshape((5,) + (1,) * x0g.ndim)
    oy = np.array([0.0, 0.0, 0.0, delta0, -delta0]).reshape((5,) + (1,) * x0g.ndim)
    tau = np.full(x0g.shape, np.nan)
    status = np.zeros(x0g.shape, dtype=int)
    sep_prev = np.full(x0g.shape, float(delta0))
    el_prev = 0.0
    for k, t, x, y, alive, _ in _march(x0g + ox, y0g + oy, t0, T, dt, vel):
        if k == 0:
            status[~alive.all(axis=0)] = 2
            continue
        el = abs(t - t0)
        sep = np.hypot(x[1:] - x[0], y[1:] - y[0]).max(axis=0)
        pending = status == 0
        ok = alive.all(axis=0)
        hit = pending & ok & (sep >= deltaf)
        with np.errstate(invalid="ignore", divide="ignore"):
            w = np.clip((deltaf - sep_prev) / (sep - sep_prev), 0.0, 1.0)
        tau[hit] = (el_prev + w * (el - el_prev))[hit]
        status[hit] = 1
        status[pending & ~ok] = 2
        sep_prev, el_prev = sep, el
        if not (status == 0).any():
            break
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(status == 1, np.log(deltaf / delta0) / tau, 0.0)
    lam[status == 2] = np.nan
    return lam, tau, status


# ---------------------------------------------------------------------------
# LAVD and eddy extraction
# ---------------------------------------------------------------------------
def lavd(x0g, y0g, t0, T, dt, vel, vort, mean_vort=0.0, return_exit=False):
    """Lagrangian-averaged vorticity deviation (Haller et al. 2016).

    ``LAVD(x0) = int_{t0}^{t0+T} |omega(x(s), s) - mean_vort(s)| ds``
    accumulated with the trapezoidal rule along RK4 trajectories.

    Parameters
    ----------
    x0g, y0g : arrays of release positions [L]
    t0, T, dt : release time, signed integration time, step [T]
    vel : velocity function;  vort : callable ``vort(x, y, t)`` [1/T]
    mean_vort : float or callable ``mean_vort(t)``, spatial-mean vorticity [1/T]
    return_exit : also return the per-particle integration time ``tau``

    Returns
    -------
    L : array like x0g (dimensionless: [1/T] x [T]); NaN for particles that
    left the valid field before |T|.
    """
    mv = mean_vort if callable(mean_vort) else (lambda t: mean_vort)
    L = prev = None
    h = abs(_time_steps(T, dt)[1])
    for k, t, x, y, alive, tau in _march(x0g, y0g, t0, T, dt, vel):
        with np.errstate(invalid="ignore"):
            dev = np.abs(vort(x, y, t) - mv(t))
        if k == 0:
            L = np.zeros(x.shape)
        else:
            L = L + np.where(alive, 0.5 * (prev + dev) * h, 0.0)
        prev = dev
    L = np.where(alive & np.isfinite(L), L, np.nan)
    return (L, tau) if return_exit else L


def _polygon_area(p):
    """Unsigned shoelace area of a closed polygon (N, 2)."""
    return 0.5 * abs(np.sum(p[:-1, 0] * p[1:, 1] - p[1:, 0] * p[:-1, 1]))


def extract_eddies(x, y, F, peak_quantile=0.9, box=None, n_levels=100, cd_max=0.01,
                   min_area=0.0, periodic_x=None, stop_at_first_failure=False):
    """Rotationally coherent eddies as outermost near-convex contours of ``F``.

    Around each local maximum of ``F`` (typically LAVD) the contour levels are
    scanned downward from the peak. A closed contour that encloses the peak and
    no other peak is accepted if its convexity deficiency
    ``CD = (A_hull - A) / A <= cd_max``; the outermost accepted contour is the
    eddy boundary.

    Parameters
    ----------
    x, y : 1-D arrays (nx,), (ny,) [L];  F : array (ny, nx), NaN allowed
    peak_quantile : peaks must exceed this quantile of F
    box : float [L], half-width of the neighbourhood in which a peak must be
        the maximum (default 10 grid cells). Sets the minimum eddy spacing.
    n_levels : number of contour levels between min(F) and max(F)
    cd_max : largest admissible convexity deficiency
    min_area : discard eddies smaller than this [L^2]
    periodic_x : float, optional. Period in x; ``F`` is tiled once so eddies
        straddling the seam are found. ``x`` must cover one period without
        repeating the end point.
    stop_at_first_failure : if True the scan stops at the first contour that
        violates ``cd_max`` (strict rule); if False it continues outward and
        keeps the outermost admissible one (tolerant to small dents in F).

    Returns
    -------
    list of dict with keys ``xc, yc`` (centre = peak), ``peak``, ``level``,
    ``boundary`` (N, 2), ``area`` [L^2], ``r_eq`` [L], ``cd``; sorted by x.
    """
    import contourpy
    from matplotlib.path import Path
    from scipy.ndimage import maximum_filter
    from scipy.spatial import ConvexHull

    x, y, F = np.asarray(x, float), np.asarray(y, float), np.asarray(F, float)
    x_start = x[0]
    if periodic_x is not None:
        x = np.concatenate([x, x + periodic_x])
        F = np.concatenate([F, F], axis=1)
    dx, dy = x[1] - x[0], y[1] - y[0]
    box = 10 * max(dx, dy) if box is None else box
    size = (2 * max(int(round(box / dy)), 1) + 1, 2 * max(int(round(box / dx)), 1) + 1)

    Ff = np.where(np.isfinite(F), F, -np.inf)
    is_peak = (Ff == maximum_filter(Ff, size=size, mode="nearest")) \
        & (Ff >= np.nanquantile(F, peak_quantile))
    pj, pi = np.nonzero(is_peak)
    peaks = np.column_stack([x[pi], y[pj]])
    pvals = F[pj, pi]

    cg = contourpy.contour_generator(x, y, np.ma.masked_invalid(F),
                                     line_type=contourpy.LineType.Separate)
    levels = np.linspace(np.nanmax(F), np.nanmin(F), n_levels + 2)[1:-1]
    best = [None] * len(peaks)
    active = np.ones(len(peaks), bool)
    for lev in levels:
        todo = np.nonzero(active & (pvals > lev))[0]
        if len(todo) == 0:
            continue
        polys = [p for p in cg.lines(lev)
                 if len(p) >= 4 and np.allclose(p[0], p[-1])]
        inside = [Path(p).contains_points(peaks) for p in polys]
        for n in todo:
            mine = [(p, m) for p, m in zip(polys, inside) if m[n]]
            if not mine:                       # contour opened (edge, gap, land)
                active[n] = False
                continue
            p, m = min(mine, key=lambda pm: _polygon_area(pm[0]))
            if m.sum() > 1:                    # would merge with another eddy
                active[n] = False
                continue
            A = _polygon_area(p)
            cd = (ConvexHull(p[:-1]).volume - A) / A if A > 0 else np.inf
            if cd <= cd_max:
                best[n] = dict(xc=peaks[n, 0], yc=peaks[n, 1], peak=pvals[n],
                               level=lev, boundary=p, area=A,
                               r_eq=np.sqrt(A / np.pi), cd=cd)
            elif stop_at_first_failure:
                active[n] = False

    eddies = [e for e in best if e is not None and e["area"] >= min_area]
    if periodic_x is not None:                 # keep one copy of each eddy
        keep = []
        for e in eddies:
            if not (x_start + 0.5 * periodic_x <= e["xc"] < x_start + 1.5 * periodic_x):
                continue
            if e["xc"] >= x_start + periodic_x:
                e["xc"] -= periodic_x
                e["boundary"] = e["boundary"] - [periodic_x, 0.0]
            keep.append(e)
        eddies = keep
    return sorted(eddies, key=lambda e: e["xc"])


# ---------------------------------------------------------------------------
# Eulerian diagnostics
# ---------------------------------------------------------------------------
def vorticity(u, v, dx, dy):
    """Relative vorticity dv/dx - du/dy by centred differences.

    ``u``, ``v`` : arrays (..., ny, nx) in [L]/[T]; ``dx``, ``dy`` in [L].
    Returns an array of the same shape in [1/T].
    """
    return np.gradient(v, dx, axis=-1) - np.gradient(u, dy, axis=-2)


def okubo_weiss(u, v, dx, dy):
    """Okubo-Weiss parameter ``W = s_n^2 + s_s^2 - omega^2``.

    ``u``, ``v`` : arrays (..., ny, nx) in [L]/[T]; ``dx``, ``dy`` in [L].

    Returns
    -------
    W [1/T^2], omega [1/T], s_n (normal strain) [1/T], s_s (shear strain) [1/T]
    """
    ux, uy = np.gradient(u, dx, axis=-1), np.gradient(u, dy, axis=-2)
    vx, vy = np.gradient(v, dx, axis=-1), np.gradient(v, dy, axis=-2)
    sn, ss, om = ux - vy, vx + uy, vx - uy
    return sn ** 2 + ss ** 2 - om ** 2, om, sn, ss


def stagnation_points(vel, x, y, t, periodic_x=None, speed_frac=0.2):
    """Instantaneous stagnation points (u = v = 0) of a velocity field.

    Candidates are the local minima of the speed on the grid ``x``, ``y``; each
    is refined with a root finder and classified with the Okubo-Weiss parameter
    of the local velocity gradient: ``"elliptic"`` (W < 0, an eddy centre of
    the frozen field) or ``"hyperbolic"`` (W > 0, a saddle).

    Parameters
    ----------
    vel : velocity function;  x, y : 1-D search grids [L];  t : time [T]
    periodic_x : float, optional. Period in x (results wrapped into [x[0], x[0]+L))
    speed_frac : candidates must be slower than this fraction of the max speed

    Returns
    -------
    list of dict with keys ``x``, ``y`` [L], ``kind``, ``W`` [1/T^2], sorted by x.
    These are properties of one snapshot in one frame: useful to choose release
    points, but they are not the Lagrangian eddy centres (see LAVD).
    """
    from scipy.ndimage import minimum_filter
    from scipy.optimize import root

    x, y = np.asarray(x, float), np.asarray(y, float)
    X, Y = np.meshgrid(x, y)
    u, v = vel(X, Y, t)
    speed = np.hypot(u, v)
    mode = ("nearest", "wrap" if periodic_x is not None else "nearest")
    cand = (speed == minimum_filter(speed, size=9, mode=mode)) & (speed < speed_frac * np.nanmax(speed))
    h = 1e-3 * min(x[1] - x[0], y[1] - y[0])
    off = np.array([-h, 0.0, h])
    out = []
    for j, i in zip(*np.nonzero(cand)):
        sol = root(lambda p: vel(p[0], p[1], t), [x[i], y[j]])
        xs, ys = sol.x
        if not sol.success or not (y[0] <= ys <= y[-1]):
            continue
        if periodic_x is not None:
            xs = x[0] + (xs - x[0]) % periodic_x
        elif not (x[0] <= xs <= x[-1]):
            continue

        def dist(o):
            dx = xs - o["x"]
            if periodic_x is not None:
                dx = (dx + 0.5 * periodic_x) % periodic_x - 0.5 * periodic_x
            return np.hypot(dx, ys - o["y"])

        if any(dist(o) < 2 * max(x[1] - x[0], y[1] - y[0]) for o in out):
            continue
        W = okubo_weiss(*vel(*np.meshgrid(xs + off, ys + off), t), h, h)[0][1, 1]
        out.append(dict(x=float(xs), y=float(ys), kind="elliptic" if W < 0 else "hyperbolic", W=float(W)))
    return sorted(out, key=lambda o: o["x"])


# ---------------------------------------------------------------------------
# Observed / modelled surface currents: loading, gaps, tides
# ---------------------------------------------------------------------------
_U_STD = ("eastward_sea_water_velocity", "surface_eastward_sea_water_velocity", "eastward_sea_surface_velocity")
_V_STD = ("northward_sea_water_velocity", "surface_northward_sea_water_velocity", "northward_sea_surface_velocity")
_U_NAMES = ("U", "u", "EWCT", "u_we", "uo")
_V_NAMES = ("V", "v", "NSCT", "v_sn", "vo")
_LON_NAMES = ("lon", "LON", "longitude", "LONGITUDE", "NbLongitudes", "lon_uv", "lon_rho")
_LAT_NAMES = ("lat", "LAT", "latitude", "LATITUDE", "NbLatitudes", "lat_uv", "lat_rho")


def _find(ds, std_names, names, what):
    for v in ds.variables:
        if ds[v].attrs.get("standard_name") in std_names:
            return v
    for n in names:
        if n in ds.variables:
            return n
    raise KeyError(f"no {what} variable found (looked for standard_name {std_names} and names {names})")


def load_surface_currents(path, t_start=None, t_end=None, good_flags=(1, 2), pad_hours=0.0):
    """Read gridded surface currents (HF radar or model surface layer) from a NetCDF file.

    Variables are recognised by CF ``standard_name`` first, then by common names;
    longitude/latitude may be 1-D or 2-D (rectilinear). cm/s are converted to m/s,
    a length-1 depth dimension is dropped, and where the velocity declares CF
    ``ancillary_variables`` with quality flags only ``good_flags`` are kept.

    Parameters
    ----------
    path : str, file name (or OPeNDAP URL)
    t_start, t_end : str or datetime64, optional. Period to keep (``t_end`` inclusive)
    good_flags : flag values kept; None = ignore quality flags
    pad_hours : extra hours loaded on each side (needed before a tidal filter)

    Returns
    -------
    xarray.Dataset with dims (time, lat, lon), variables ``u``, ``v`` [m/s] (NaN = no
    data) and attrs ``source``, ``period``, ``n_flagged``.
    """
    import xarray as xr
    raw = xr.open_dataset(path)
    un, vn = _find(raw, _U_STD, _U_NAMES, "eastward velocity"), _find(raw, _V_STD, _V_NAMES, "northward velocity")
    lon_n = next(n for n in _LON_NAMES if n in raw.variables)
    lat_n = next(n for n in _LAT_NAMES if n in raw.variables)
    tn = next(d for d in raw[un].dims if np.issubdtype(raw[d].dtype, np.datetime64)) if any(
        d in raw.variables and np.issubdtype(raw[d].dtype, np.datetime64) for d in raw[un].dims) else None
    if tn is None:
        raise ValueError("the velocity has no datetime time coordinate")
    time = raw[tn].dt.round("s").values
    if t_start is not None or t_end is not None:
        pad = np.timedelta64(int(round(pad_hours * 3600)), "s")
        lo = time[0] if t_start is None else np.datetime64(t_start) - pad
        hi = time[-1] if t_end is None else np.datetime64(t_end) + pad
        keep = (time >= lo) & (time <= hi)
    else:
        keep = np.ones(len(time), bool)
    raw = raw.isel({tn: keep})
    lon, lat = raw[lon_n].values, raw[lat_n].values
    ydim, xdim = [d for d in raw[un].dims if d != tn and raw[un].sizes[d] > 1][-2:]
    if lon.ndim == 2:                                    # 2-D but rectilinear
        lon_ax = raw[lon_n].dims.index(xdim)
        lon = lon.take(0, axis=1 - lon_ax)
        lat = lat.take(0, axis=raw[lat_n].dims.index(xdim))
    out = {}
    n_flagged = 0
    for name, var in (("u", un), ("v", vn)):
        da = raw[var].squeeze(drop=True).transpose(tn, ydim, xdim)
        a = np.asarray(da.values, float)
        units = str(da.attrs.get("units", "")).replace(" ", "").lower()
        if units in ("cm/s", "cms-1", "cm.s-1", "centimeter/second"):
            a = a / 100.0
        anc = str(da.attrs.get("ancillary_variables", "")).split()
        if good_flags is not None and anc and anc[0] in raw.variables:
            ok = np.isin(raw[anc[0]].squeeze(drop=True).transpose(tn, ydim, xdim).values, good_flags)
            n_flagged += int((np.isfinite(a) & ~ok).sum())
            a = np.where(ok, a, np.nan)
        out[name] = (("time", "lat", "lon"), a)
    ds = xr.Dataset(out, coords={"time": time[keep], "lat": lat, "lon": lon})
    ds.attrs.update(source=str(path), period=f"{str(ds.time.values[0])[:16]} to {str(ds.time.values[-1])[:16]}",
                    n_flagged=n_flagged, u_name=un, v_name=vn)
    return ds


def fill_gaps(dk, min_coverage=0.7, max_gap=6.0):
    """Make a gappy velocity dataset continuous inside its reliable footprint.

    1. footprint = cells with data at least ``min_coverage`` of the time; the rest is NaN;
    2. gaps of at most ``max_gap`` [T] are filled by linear interpolation in time;
    3. what remains is filled, time step by time step, with the mean of the valid
       neighbours (repeated until the footprint is full);
    4. times when the whole field is missing for longer are interpolated in time
       without limit (last resort).

    This is a simple teaching tool. Operational gap filling uses methods that respect
    the dynamics and the measurement errors (2DVar, OMA, DINEOF).

    Parameters
    ----------
    dk : Dataset with dims (time, y, x), variables ``u``, ``v`` and a numeric ``time`` [T]

    Returns
    -------
    filled : Dataset like ``dk``;  footprint : bool array (y, x);
    stats : dict with the valid fraction inside the footprint after each step
    """
    from scipy.ndimage import uniform_filter
    good = np.isfinite(dk.u.values) & np.isfinite(dk.v.values)
    foot = good.mean(axis=0) >= min_coverage
    tt = np.asarray(dk.time.values, float)
    cells = list(zip(*np.nonzero(foot)))
    arr = {n: np.where(foot, dk[n].values, np.nan).astype(float) for n in ("u", "v")}
    frac = lambda: float(np.isfinite(arr["u"])[:, foot].mean()) if foot.any() else 0.0
    stats = {"raw": frac()}
    for a in arr.values():                               # 2. short gaps, in time
        for j, i in cells:
            ok = np.isfinite(a[:, j, i])
            if ok.all() or ok.sum() < 2:
                continue
            k = np.nonzero(ok)[0]
            nxt = np.searchsorted(k, np.arange(len(tt)))
            inside = (nxt > 0) & (nxt < len(k)) & ~ok
            gap = np.zeros(len(tt))
            gap[inside] = tt[k[np.minimum(nxt, len(k) - 1)]][inside] - tt[k[np.maximum(nxt - 1, 0)]][inside] - (tt[1] - tt[0])
            fill = inside & (gap <= max_gap)
            a[fill, j, i] = np.interp(tt[fill], tt[ok], a[ok, j, i])
    stats["time"] = frac()
    for n, a in arr.items():                             # 3. neighbours, in space
        for _ in range(60):
            nan = np.isnan(a) & foot
            if not nan.any():
                break
            w = uniform_filter(np.isfinite(a).astype(float), size=(1, 3, 3), mode="constant")
            sm = uniform_filter(np.nan_to_num(a), size=(1, 3, 3), mode="constant")
            with np.errstate(invalid="ignore", divide="ignore"):
                a = np.where(nan & (w > 0), sm / w, a)
        arr[n] = a
    stats["space"] = frac()
    for a in arr.values():                               # 4. long outages
        for j, i in cells:
            ok = np.isfinite(a[:, j, i])
            if not ok.all() and ok.any():
                a[~ok, j, i] = np.interp(tt[~ok], tt[ok], a[ok, j, i])
    stats["final"] = frac()
    out = dk.copy()
    for n, a in arr.items():
        out[n] = (dk[n].dims, a)
    return out, foot, stats


def godin_filter(ds, time="time"):
    """Godin (1972) tide-killer low-pass filter for hourly data: running means of 24, 24, 25 h.

    It removes the diurnal and semidiurnal tides (and most of the inertial motion at
    mid latitudes) and returns the subtidal, "residual" flow. The first and last
    35 hours cannot be filtered and become NaN: load the data with a padding of at
    least 36 h and crop afterwards.

    Parameters
    ----------
    ds : Dataset or DataArray sampled every hour along ``time``

    Returns
    -------
    Same type and shape as ``ds``.
    """
    out = ds
    for n in (24, 24, 25):
        out = out.rolling({time: n}, center=True, min_periods=n).mean()
    return out


def ow_eddies(x, y, u, v, min_cells=4, threshold=0.2):
    """Okubo-Weiss eddies of one velocity snapshot: closed contours of W = -threshold * sigma_W.

    Parameters
    ----------
    x, y : 1-D grids [L];  u, v : arrays (ny, nx) [L]/[T], NaN where undefined

    Returns
    -------
    W (ny, nx) [1/T^2];  level (float);  eddies : list of dict with ``boundary`` (N, 2),
    ``area``, ``r_eq``, ``xc``, ``yc`` (position of the W minimum), ``cyclonic`` (bool);
    n_regions : number of connected regions below the level (those cut by the edge of
    the data have no closed boundary and are not in ``eddies``).
    """
    import contourpy
    from matplotlib.path import Path
    from scipy import ndimage
    W, om, _, _ = okubo_weiss(u, v, x[1] - x[0], y[1] - y[0])
    level = -threshold * np.nanstd(W)
    n_regions = ndimage.label(np.nan_to_num(W, nan=np.inf) < level)[1]
    cg = contourpy.contour_generator(x, y, np.ma.masked_invalid(W), line_type=contourpy.LineType.Separate)
    X, Y = np.meshgrid(x, y)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    eddies = []
    for p in cg.lines(level):
        if len(p) < 4 or not np.allclose(p[0], p[-1]):
            continue
        inside = Path(p).contains_points(pts).reshape(W.shape) & np.isfinite(W)
        if inside.sum() < min_cells or np.nanmean(W[inside]) > level:
            continue
        j, i = np.unravel_index(np.argmin(np.where(inside, W, np.inf)), W.shape)
        A = _polygon_area(p)
        eddies.append(dict(boundary=p, area=A, r_eq=np.sqrt(A / np.pi), xc=x[i], yc=y[j], cyclonic=bool(om[j, i] > 0)))
    return W, level, eddies, n_regions


def advect_curve(boundary, t0, T, dt, vel, n=400):
    """Advect a closed material curve and measure how it deforms.

    Parameters
    ----------
    boundary : array (N, 2), closed curve [L];  t0, T, dt : as in :func:`advect`
    n : number of points the curve is resampled to

    Returns
    -------
    dict with ``initial`` and ``final`` (n, 2) curves, ``ratio`` (final / initial
    perimeter), ``stay`` (fraction of points still in the valid field after |T|) and
    ``t_common`` (time [T] of ``final``: the last time all points were valid).
    """
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(boundary, axis=0).T))])
    si = np.linspace(0, s[-1], n)
    bx, by = np.interp(si, s, boundary[:, 0]), np.interp(si, s, boundary[:, 1])
    X, Y, t, tau = advect(bx, by, t0, T, dt, vel, save_every=2, return_exit=True)
    k = max(int(np.searchsorted(np.abs(t - t0), tau.min() + 1e-9, side="right")) - 1, 0)
    per = lambda a, b: np.hypot(np.diff(a), np.diff(b)).sum()
    return dict(initial=np.column_stack([bx, by]), final=np.column_stack([X[k], Y[k]]),
                ratio=per(X[k], Y[k]) / per(bx, by), stay=float((tau >= abs(T) - 1e-9).mean()),
                t_common=float(abs(t[k] - t0)))


# ---------------------------------------------------------------------------
# Trajectory skill
# ---------------------------------------------------------------------------
def liu_weisberg_skill(obs_xy, sim_xy, n=1.0):
    """Liu & Weisberg (2011) normalised cumulative separation skill score.

    ``c = sum_i d_i / sum_i l_oi`` with ``d_i`` the observed-simulated
    separation and ``l_oi`` the observed path length up to time i;
    ``ss = 1 - c/n`` if ``c <= n`` else 0.

    Parameters
    ----------
    obs_xy, sim_xy : arrays (nt, 2), positions [L] at the same times, starting
        from the same point
    n : tolerance threshold (1 = standard)

    Returns
    -------
    ss : float in [0, 1];  c : float
    """
    obs, sim = np.asarray(obs_xy, float), np.asarray(sim_xy, float)
    d = np.hypot(*(obs[1:] - sim[1:]).T)
    lo = np.cumsum(np.hypot(*np.diff(obs, axis=0).T))
    ok = np.isfinite(d) & np.isfinite(lo)
    if not ok.any() or lo[ok].sum() == 0:
        return np.nan, np.nan
    c = d[ok].sum() / lo[ok].sum()
    return (1.0 - c / n if c <= n else 0.0), c


# ---------------------------------------------------------------------------
# Particle advection (backend 2: OceanParcels)
# ---------------------------------------------------------------------------
def advect_parcels(x0, y0, t0, T, dt, ds, save_every=None, return_exit=False,
                   var_u="u", var_v="v", length_m=1000.0, time_s=86400.0, scale=1.0):
    """Same contract as :func:`advect`, integrated by OceanParcels (v4, RK4).

    Parcels works in SI units on a flat mesh, so the gridded field is converted
    on the way in and the trajectories on the way out.

    Parameters
    ----------
    x0, y0, t0, T, dt, save_every, return_exit : as in :func:`advect`
    ds : xarray.Dataset with dims (time, y, x); ``x``, ``y`` in [L], numeric
        ``time`` in [T], velocities ``var_u``, ``var_v`` in [L]/[T] after
        multiplication by ``scale``
    length_m : metres per [L] (1000 for km)
    time_s : seconds per [T] (86400 for days, 3600 for hours)

    Returns
    -------
    Same as :func:`advect`. Particles leaving the grid (or meeting NaN) are
    frozen at their last valid position.
    """
    import logging
    import warnings
    import parcels
    import xarray as xr
    from parcels._core.particle import get_default_particle   # float64 positions

    shape = np.broadcast(x0, y0).shape
    xs = np.broadcast_to(np.asarray(x0, float), shape).ravel() * length_m
    ys = np.broadcast_to(np.asarray(y0, float), shape).ravel() * length_m
    n, h = _time_steps(T, dt)

    # SI copy of the field; parcels reads NaN as zero velocity, so NaN nodes
    # are zero-filled and tracked in a separate "bad" field instead
    d = ds.transpose("time", "y", "x")
    u = np.asarray(d[var_u].values, float) * scale * length_m / time_s
    v = np.asarray(d[var_v].values, float) * scale * length_m / time_s
    bad = ~(np.isfinite(u) & np.isfinite(v))
    tg = np.asarray(d["time"].values, float)
    xg, yg = np.asarray(d["x"].values, float) * length_m, np.asarray(d["y"].values, float) * length_m
    coords = {"time": np.round((tg - tg[0]) * time_s * 1e9).astype("timedelta64[ns]"),
              "lat": yg, "lon": xg}

    def da(a):
        return xr.DataArray(a, dims=("time", "lat", "lon"), coords=coords)

    level = parcels.logger.level
    parcels.logger.setLevel(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sg = parcels.convert.copernicusmarine_to_sgrid(
                fields={"U": da(np.where(bad, 0.0, u)), "V": da(np.where(bad, 0.0, v)),
                        "bad": da(bad.astype(float))})
            fieldset = parcels.FieldSet.from_sgrid_conventions(sg, mesh="flat")
        xlo, xhi, ylo, yhi = xg[0], xg[-1], yg[0], yg[-1]
        tspan = (tg[-1] - tg[0]) * time_s

        def FreezeInvalid(particles, fieldset):
            """Freeze particles whose step starts or ends outside the valid field."""
            xn, yn = particles.x + particles.dx, particles.y + particles.dy
            tn = np.clip(particles.t + particles.dt, 0.0, tspan)
            ok = (xn >= xlo) & (xn <= xhi) & (yn >= ylo) & (yn <= yhi)
            ok &= (particles.t + particles.dt >= -1e-6) & (particles.t + particles.dt <= tspan + 1e-6)
            ok &= np.isfinite(xn) & np.isfinite(yn) & (particles.state < 50)
            ok &= fieldset.bad[particles] == 0
            ok &= fieldset.bad[tn, particles.z, np.clip(np.nan_to_num(yn), ylo, yhi),
                               np.clip(np.nan_to_num(xn), xlo, xhi), particles] == 0
            # sampling errors (out of bounds) become a frozen particle, not a crash
            particles.state = np.where(particles.state >= 50, 0, particles.state)
            particles.alive = particles.alive * ok
            particles.dx = np.where(particles.alive > 0, particles.dx, 0.0)
            particles.dy = np.where(particles.alive > 0, particles.dy, 0.0)
            particles.tau += np.abs(particles.dt) * particles.alive

        pclass = get_default_particle(np.float64).add_variable(
            [parcels.Variable("alive", dtype=np.float64, initial=1.0),
             parcels.Variable("tau", dtype=np.float64, initial=0.0)])
        t_rel = np.timedelta64(int(round((t0 - tg[0]) * time_s * 1e9)), "ns")
        pset = parcels.ParticleSet(fieldset, pclass=pclass, x=xs, y=ys,
                                   t=np.full(xs.size, t_rel))
        inside = (xs >= xlo) & (xs <= xhi) & (ys >= ylo) & (ys <= yhi)
        pset.alive[:] = inside.astype(float)

        chunk = n if save_every is None else save_every
        X, Y, ts = [np.array(pset.x)], [np.array(pset.y)], [t0]
        done = 0
        while done < n:
            m = min(chunk, n - done)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pset.execute([parcels.kernels.AdvectionRK4, FreezeInvalid],
                             dt=h * time_s, runtime=m * abs(h) * time_s,
                             verbose_progress=False)
            done += m
            X.append(np.array(pset.x)), Y.append(np.array(pset.y)), ts.append(t0 + done * h)
        tau = np.minimum(np.array(pset.tau) / time_s, abs(T)).reshape(shape)
    finally:
        parcels.logger.setLevel(level)

    X = np.array(X).reshape((-1,) + shape) / length_m
    Y = np.array(Y).reshape((-1,) + shape) / length_m
    out = (X[-1], Y[-1]) if save_every is None else (X, Y, np.array(ts))
    return out + (tau,) if return_exit else out


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def plot_field(ax, x, y, F, kind="ftle_fwd", label=None, title=None, robust=True,
               colorbar=True, **kw):
    """``pcolormesh`` with the course colormap convention (see ``CMAPS``).

    ``kind`` is one of ``CMAPS``. Diverging kinds (vorticity, okubo_weiss) get
    limits symmetric about zero. With ``robust`` the limits are the 1st/99th
    percentiles. Returns the QuadMesh.
    """
    import matplotlib.pyplot as plt
    F = np.asarray(F, float)
    kw.setdefault("cmap", CMAPS.get(kind, kind))
    kw.setdefault("shading", "auto")
    if "vmin" not in kw and "vmax" not in kw and np.isfinite(F).any():
        lo, hi = np.nanpercentile(F, [1, 99]) if robust else (np.nanmin(F), np.nanmax(F))
        if kind in ("vorticity", "okubo_weiss"):
            m = max(abs(lo), abs(hi))
            kw.update(vmin=-m, vmax=m)
        else:
            kw.update(vmin=max(lo, 0.0) if np.nanmin(F) >= 0 else lo, vmax=hi)
    pm = ax.pcolormesh(x, y, F, **kw)
    if colorbar:
        plt.colorbar(pm, ax=ax, label=label, pad=0.02)
    if title:
        ax.set_title(title)
    return pm


def plot_eddies(ax, eddies, color="k", lw=1.5, centers=True, label=None, shift_x=0.0):
    """Draw eddy boundaries (and centres) from :func:`extract_eddies`."""
    for n, e in enumerate(eddies):
        b = e["boundary"]
        ax.plot(b[:, 0] + shift_x, b[:, 1], color=color, lw=lw,
                label=label if n == 0 else None)
        if centers:
            ax.plot(e["xc"] + shift_x, e["yc"], marker="+", color=color, ms=8)


def coastline(lon_min, lon_max, lat_min, lat_max, scale="f", cache=None):
    """High-resolution coastline (GSHHS land polygons) clipped to a lon/lat box.

    Parameters
    ----------
    lon_min, lon_max, lat_min, lat_max : box [degrees]
    scale : GSHHS resolution: "c", "l", "i", "h" (high, ~200 m) or "f" (full, ~50 m)
    cache : str, optional. Path of a ``.npz`` file: read if it exists, otherwise
        written after the first extraction, so that classrooms can work offline
        (cartopy downloads the GSHHS shapefile on first use and needs internet).

    Returns
    -------
    list of (N, 2) arrays of closed land rings, columns lon, lat [degrees].
    """
    import os
    if cache is not None and os.path.exists(cache):
        with np.load(cache) as f:
            return np.split(f["lonlat"], f["offsets"])
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import box
    bb = box(lon_min, lat_min, lon_max, lat_max)
    rings = []
    for g in shpreader.Reader(shpreader.gshhs(scale, 1)).geometries():
        if not g.intersects(bb):
            continue
        g = g.intersection(bb)
        for poly in getattr(g, "geoms", [g]):
            if poly.geom_type == "Polygon" and not poly.is_empty:
                rings.append(np.asarray(poly.exterior.coords, float)[:, :2])
    if cache is not None:
        np.savez_compressed(cache, lonlat=np.concatenate(rings),
                            offsets=np.cumsum([len(r) for r in rings])[:-1])
    return rings


def plot_coastline(ax, rings, lon0=None, lat0=None, facecolor="0.85", edgecolor="0.25", lw=0.6, zorder=3):
    """Draw land polygons from :func:`coastline`.

    With ``lon0``, ``lat0`` the rings are projected to the local km frame of
    :func:`lonlat_to_km` (use the ``lon0``/``lat0`` attributes of the km
    dataset); otherwise they are drawn in degrees. Axis limits are left
    unchanged, so set them before or after as needed.
    """
    from matplotlib.collections import PolyCollection
    polys = [np.column_stack(lonlat_to_km(r[:, 0], r[:, 1], lon0, lat0)) if lon0 is not None else r
             for r in rings]
    ax.add_collection(PolyCollection(polys, facecolors=facecolor, edgecolors=edgecolor,
                                     linewidths=lw, zorder=zorder), autolim=False)


class _quiet:
    """Silence numpy's 'Mean of empty slice' when no particle survives."""

    def __enter__(self):
        import warnings
        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)
