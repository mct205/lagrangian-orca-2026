"""Tests for lagkit. Run with ``pytest`` from the repository root."""
import numpy as np
import pytest

import lagkit as lk


@pytest.fixture(scope="module")
def bj():
    return lk.BickleyJet()


def uniform(x, y, t):
    return np.full_like(x, 3.0), np.full_like(y, -1.0)


def saddle(alpha):
    return lambda x, y, t: (alpha * x, -alpha * y)


def solid_body(omega):
    return lambda x, y, t: (-omega * y, omega * x)


# -- analytic flow ----------------------------------------------------------
def test_bickley_divergence_and_vorticity(bj):
    rng = np.random.default_rng(42)
    x = rng.uniform(0, bj.Lx, 500)
    y = rng.uniform(-3000, 3000, 500)
    t, h = 7.3, 1e-2
    ux = (bj.velocity(x + h, y, t)[0] - bj.velocity(x - h, y, t)[0]) / (2 * h)
    uy = (bj.velocity(x, y + h, t)[0] - bj.velocity(x, y - h, t)[0]) / (2 * h)
    vx = (bj.velocity(x + h, y, t)[1] - bj.velocity(x - h, y, t)[1]) / (2 * h)
    vy = (bj.velocity(x, y + h, t)[1] - bj.velocity(x, y - h, t)[1]) / (2 * h)
    scale = bj.U0 / bj.L0
    assert np.abs(ux + vy).max() / scale < 1e-8
    assert np.abs(vx - uy - bj.vorticity(x, y, t)).max() / scale < 1e-8


def test_bickley_periodic_and_array_time(bj):
    x, y = np.array([100.0, 5000.0]), np.array([300.0, -900.0])
    np.testing.assert_allclose(bj.velocity(x, y, 3.0), bj.velocity(x + bj.Lx, y, 3.0), atol=1e-8)
    u, v = bj.velocity(x, y, np.array([1.0, 2.0]))
    assert u[1] == pytest.approx(bj.velocity(x[1], y[1], 2.0)[0])


def test_bickley_custom_speeds_and_stagnation_points(bj):
    b = lk.BickleyJet(U0=3000.0, L0=2000.0, eps=(0.0, 0.2, 0.3), c2=0.3, c3=0.5)
    assert b.c[2] == pytest.approx(1500.0) and b.c[1] == pytest.approx(900.0)
    assert lk.BickleyJet(c1=0.1).c[0] == pytest.approx(0.1 * bj.U0)
    x = np.linspace(0, bj.Lx, 400, endpoint=False)
    y = np.linspace(-3000, 3000, 241)
    pts = lk.stagnation_points(bj.velocity, x, y, 0.0, periodic_x=bj.Lx)
    kinds = [p["kind"] for p in pts]
    assert kinds.count("elliptic") == 6 and kinds.count("hyperbolic") == 6
    for p in pts:
        assert np.hypot(*bj.velocity(p["x"], p["y"], 0.0)) < 1e-6 * bj.U0
    saddle_pts = lk.stagnation_points(saddle(0.3), np.linspace(-1, 1, 41), np.linspace(-1, 1, 41), 0.0)
    assert len(saddle_pts) == 1 and saddle_pts[0]["kind"] == "hyperbolic"


# -- advection --------------------------------------------------------------
def test_advect_solid_body_forward_backward():
    om = 0.7
    x0, y0 = np.array([[1.0, 0.0], [0.5, -2.0]]), np.array([[0.0, 1.0], [0.5, 1.0]])
    T = 2 * np.pi / om
    xf, yf = lk.advect(x0, y0, 0.0, T, 0.01, solid_body(om))
    np.testing.assert_allclose(xf, x0, atol=1e-6)
    np.testing.assert_allclose(yf, y0, atol=1e-6)
    xb, yb = lk.advect(xf, yf, T, -T, 0.01, solid_body(om))
    np.testing.assert_allclose(xb, x0, atol=1e-6)
    X, Y, t = lk.advect(x0, y0, 0.0, 1.0, 0.01, solid_body(om), save_every=30)
    assert X.shape == (5, 2, 2) and t[-1] == pytest.approx(1.0)


def test_interpolator_and_exit():
    import xarray as xr
    x, y, t = np.arange(0, 11.0), np.arange(0, 6.0), np.array([0.0, 1.0, 4.0])
    T3, Y3, X3 = np.meshgrid(t, y, x, indexing="ij")
    ds = xr.Dataset({"u": (("time", "y", "x"), 1 + 0 * X3 + 0.1 * T3),
                     "v": (("time", "y", "x"), 0.5 * Y3 - X3)},
                    coords={"time": t, "y": y, "x": x})
    vel = lk.grid_interpolator(ds, "u", "v", scale=2.0)
    u, v = vel(np.array([2.5, 10.0, 11.0]), np.array([1.25, 5.0, 1.0]), 2.5)
    np.testing.assert_allclose(u[:2], 2 * 1.25)
    np.testing.assert_allclose(v[:2], 2 * np.array([0.625 - 2.5, 2.5 - 10]))
    assert np.isnan(u[2]) and np.isnan(vel(1.0, 1.0, 4.5)[0])
    # u = 1 + 0.1 t (x2): the particle leaves through x = 10 and is frozen there
    xf, yf, tau = lk.advect(np.array([8.0, 1.0]), np.array([0.0, 0.0]), 0.0, 3.0, 0.05,
                            lambda xx, yy, tt: (vel(xx, yy, tt)[0], 0 * xx), return_exit=True)
    assert tau[0] < 1.0 and xf[0] <= 10.0 and tau[1] == pytest.approx(3.0)
    m = lk.survival_mask(np.array([0.0, 0.5, 3.0]), 0.0, tau)
    assert m[:, 1].all() and not m[-1, 0]


def test_lonlat_roundtrip():
    lon, lat = np.array([106.2, 107.9]), np.array([19.1, 20.6])
    x, y = lk.lonlat_to_km(lon, lat, 107.0, 20.0)
    assert y[1] == pytest.approx(0.6 * np.pi / 180 * 6371.0)
    lon2, lat2 = lk.km_to_lonlat(x, y, 107.0, 20.0)
    np.testing.assert_allclose([lon2, lat2], [lon, lat])


# -- FTLE / FSLE ------------------------------------------------------------
def _grid(n=41, half=1.0):
    x = np.linspace(-half, half, n)
    return x, x, *np.meshgrid(x, x)


def test_ftle_uniform_flow_is_zero():
    x, y, X, Y = _grid()
    xf, yf = lk.advect(X, Y, 0.0, 5.0, 0.1, uniform)
    ftle = lk.ftle_from_flowmap(xf, yf, x[1] - x[0], y[1] - y[0], 5.0)
    assert np.abs(ftle).max() < 1e-10


@pytest.mark.parametrize("T", [2.0, -2.0])
def test_ftle_linear_saddle_equals_alpha(T):
    alpha = 0.37
    x, y, X, Y = _grid()
    xf, yf = lk.advect(X, Y, 0.0, T, 0.02, saddle(alpha))
    ftle = lk.ftle_from_flowmap(xf, yf, x[1] - x[0], y[1] - y[0], T)
    np.testing.assert_allclose(ftle, alpha, rtol=1e-6)


def test_ftle_from_trajectories_matches_and_masks():
    alpha, T = 0.37, 2.0

    def vel(x, y, t):                       # saddle, undefined for x > 2.2
        u, v = saddle(alpha)(x, y, t)
        return np.where(x > 2.2, np.nan, u), v

    x, y, X, Y = _grid(half=2.0)
    Xs, Ys, t, tau = lk.advect(X, Y, 0.0, T, 0.02, vel, save_every=5, return_exit=True)
    ftle, tau_eff = lk.ftle_from_trajectories(Xs, Ys, t, tau, x[1] - x[0], y[1] - y[0])
    assert (tau < T).any() and np.isfinite(ftle).sum() > ftle.size // 2
    np.testing.assert_allclose(ftle[np.isfinite(ftle)], alpha, rtol=1e-5)
    assert np.isnan(ftle[:, -1]).all()      # these left almost immediately
    f2, t2 = lk.ftle_from_flowmap(Xs[-1], Ys[-1], x[1] - x[0], y[1] - y[0], T, tau=tau)
    np.testing.assert_allclose(f2[t2 >= T], alpha, rtol=1e-6)


def test_ftle_periodic_seam(bj):
    x = np.linspace(0, bj.Lx, 80, endpoint=False)
    y = np.linspace(-2500, 2500, 41)
    X, Y = np.meshgrid(x, y)
    xf, yf = lk.advect(X, Y, 0.0, 3.0, 0.05, bj)
    a = lk.ftle_from_flowmap(xf, yf, x[1] - x[0], y[1] - y[0], 3.0, periodic_x=bj.Lx)
    # reference: same grid with two halo columns on each side, no periodicity
    xe = np.concatenate([x[-2:] - bj.Lx, x, x[:2] + bj.Lx])
    Xe, Ye = np.meshgrid(xe, y)
    xfe, yfe = lk.advect(Xe, Ye, 0.0, 3.0, 0.05, bj)
    b = lk.ftle_from_flowmap(xfe, yfe, x[1] - x[0], y[1] - y[0], 3.0)[:, 2:-2]
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_fsle_map_saddle():
    alpha = 0.5
    X, Y = np.meshgrid(np.linspace(-0.2, 0.2, 5), np.linspace(-0.2, 0.2, 5))
    lam, tau, status = lk.fsle_map(X, Y, 0.0, 10.0, 0.01, 0.01, 0.16, saddle(alpha))
    assert (status == 1).all()
    np.testing.assert_allclose(lam, alpha, rtol=2e-3)
    lam0, _, st0 = lk.fsle_map(X, Y, 0.0, 10.0, 0.1, 0.01, 0.16, uniform)
    assert (st0 == 0).all() and (lam0 == 0).all()


def test_fsle_statistic_exponential_pairs():
    lam, d0 = 0.3, 1.0
    t = np.linspace(0, 20, 401)
    rng = np.random.default_rng(42)
    D = d0 * np.exp(lam * t[:, None]) * rng.uniform(0.8, 1.0, 50)[None, :]
    delta, lam_est, npairs, _ = lk.fsle_statistic(D, t, d0)
    np.testing.assert_allclose(lam_est[1:-1], lam, rtol=1e-2)
    assert npairs[1] == 50
    # pairs released at 4*d0 must not be counted in the bins below their initial separation
    D_big = 4 * d0 * np.exp(lam * t[:, None]) * np.ones((1, 30))
    _, lam2, n2, _ = lk.fsle_statistic(np.concatenate([D, D_big], axis=1), t, d0, n_max=8)
    assert n2[0] == 50 and n2[1] == 50 and n2[4] == 80
    np.testing.assert_allclose(lam2[1:6], lam, rtol=1e-2)


# -- statistics -------------------------------------------------------------
def test_absolute_dispersion_and_autocorr():
    rng = np.random.default_rng(42)
    t = np.linspace(0, 10, 101)
    u = rng.normal(0, 2.0, 2000)
    X = 5.0 * t[:, None] + u[None, :] * t[:, None]          # ballistic + mean drift
    Ax2, Ay2, n = lk.absolute_dispersion(X, 0 * X)
    np.testing.assert_allclose(Ax2, u.var() * t ** 2, rtol=1e-10, atol=1e-12)
    assert Ay2.max() == 0 and n[0] == 2000
    q = np.cos(2 * np.pi * t / 4.0)[:, None] * np.ones((1, 3))
    R = lk.lagrangian_autocorr(q, 30)
    TL, i0 = lk.integral_timescale(R, t[1] - t[0])
    assert R[0] == 1 and 0.9 <= i0 * (t[1] - t[0]) <= 1.1 and TL > 0


def test_pair_separation_minimum_image():
    D = lk.pair_separation(np.array([[1.0]]), np.array([[0.0]]), np.array([[99.0]]),
                           np.array([[0.0]]), periodic_x=100.0)
    assert D[0, 0] == pytest.approx(2.0)
    D2, K, kurt, n = lk.relative_dispersion(np.array([[1.0, 1.0], [2.0, 2.0]]), np.array([0.0, 1.0]))
    np.testing.assert_allclose(D2, [1, 4]); np.testing.assert_allclose(kurt, 1)


# -- LAVD / eddies / OW -----------------------------------------------------
def test_lavd_solid_body_rotation():
    om, T = 0.4, 5.0
    _, _, X, Y = _grid(n=11)
    L = lk.lavd(X, Y, 0.0, T, 0.05, solid_body(om), lambda x, y, t: 2 * om + 0 * x, 0.0)
    np.testing.assert_allclose(L, 2 * om * T, rtol=1e-12)
    L0 = lk.lavd(X, Y, 0.0, T, 0.05, solid_body(om), lambda x, y, t: 2 * om + 0 * x, 2 * om)
    assert np.abs(L0).max() < 1e-12


def test_extract_eddies_gaussian():
    x, y = np.linspace(-100, 100, 201), np.linspace(-60, 60, 121)
    X, Y = np.meshgrid(x, y)
    F = np.exp(-((X - 40) ** 2 + Y ** 2) / (2 * 15 ** 2)) + np.exp(-((X + 40) ** 2 + Y ** 2) / (2 * 15 ** 2))
    ed = lk.extract_eddies(x, y, F, peak_quantile=0.9, box=20, n_levels=80, cd_max=0.01)
    assert len(ed) == 2
    assert ed[0]["xc"] == pytest.approx(-40) and ed[1]["xc"] == pytest.approx(40)
    assert all(15 < e["r_eq"] < 40 and e["cd"] <= 0.01 for e in ed)


def test_bickley_six_eddies(bj):
    x = np.linspace(0, bj.Lx, 200, endpoint=False)
    y = np.linspace(-3000, 3000, 61)
    X, Y = np.meshgrid(x, y)
    L = lk.lavd(X, Y, 0.0, 30.0, 0.1, bj, bj.vorticity, 0.0)
    ed = lk.extract_eddies(x, y, L, peak_quantile=0.9, box=800, n_levels=60, cd_max=0.02,
                           min_area=np.pi * 150 ** 2, periodic_x=bj.Lx)
    assert len(ed) == 6
    assert sum(e["yc"] > 0 for e in ed) == 3


def test_okubo_weiss():
    x, y, X, Y = _grid()
    d = x[1] - x[0]
    W, om, sn, ss = lk.okubo_weiss(*solid_body(0.4)(X, Y, 0), d, d)
    np.testing.assert_allclose(W, -(0.8 ** 2)); np.testing.assert_allclose(om, 0.8)
    W, om, _, _ = lk.okubo_weiss(*saddle(0.3)(X, Y, 0), d, d)
    np.testing.assert_allclose(W, 0.6 ** 2); np.testing.assert_allclose(om, 0, atol=1e-12)
    np.testing.assert_allclose(lk.vorticity(*solid_body(0.4)(X, Y, 0), d, d), 0.8)


def test_liu_weisberg_skill():
    t = np.linspace(0, 1, 20)
    obs = np.column_stack([10 * t, 3 * t ** 2])
    assert lk.liu_weisberg_skill(obs, obs)[0] == 1.0
    assert lk.liu_weisberg_skill(obs, obs * [-1, 1])[0] == 0.0
    ss, c = lk.liu_weisberg_skill(obs, obs + np.column_stack([t, 0 * t]))
    assert 0 < ss < 1 and c == pytest.approx(1 - ss)


# -- gridded data: loading, gaps, tides --------------------------------------
def _radar_like(tmp_path, nt=120):
    import xarray as xr
    rng = np.random.default_rng(42)
    time = np.datetime64("2020-01-01T00:00") + np.arange(nt) * np.timedelta64(1, "h")
    lon, lat = np.linspace(1.0, 1.3, 7), np.linspace(40.0, 40.2, 6)
    u = 20.0 + 5 * rng.standard_normal((nt, 1, 6, 7))                 # cm/s, with a depth dimension
    qc = np.ones_like(u); qc[10:13, :, 2, 3] = 4
    ds = xr.Dataset({"EWCT": (("time", "depth", "j", "i"), u, {"units": "cm/s", "standard_name": "eastward_sea_water_velocity", "ancillary_variables": "QCflag"}),
                     "NSCT": (("time", "depth", "j", "i"), -u, {"units": "cm/s", "standard_name": "northward_sea_water_velocity", "ancillary_variables": "QCflag"}),
                     "QCflag": (("time", "depth", "j", "i"), qc),
                     "LONGITUDE": (("j", "i"), np.tile(lon, (6, 1))), "LATITUDE": (("j", "i"), np.tile(lat[:, None], (1, 7)))},
                    coords={"time": time})
    path = str(tmp_path / "radar.nc")
    ds.to_netcdf(path)
    return path, u


def test_load_surface_currents_and_fill_gaps(tmp_path):
    path, u = _radar_like(tmp_path)
    ds = lk.load_surface_currents(path, "2020-01-02", "2020-01-03T23:00")
    assert ds.u.dims == ("time", "lat", "lon") and ds.sizes["time"] == 48
    full = lk.load_surface_currents(path)
    np.testing.assert_allclose(np.nanmean(full.u.values), np.mean(np.delete(u[:, 0].ravel(), [])) / 100, rtol=0.02)
    assert np.isnan(full.u.values[10:13, 2, 3]).all() and full.attrs["n_flagged"] == 6
    assert np.isfinite(lk.load_surface_currents(path, good_flags=None).u.values).all()
    dk = lk.dataset_to_km(full)
    dk["u"].values[40:60, 0, 0] = np.nan                               # long gap in one cell
    dk["u"].values[:, 5, 6] = np.nan                                   # a cell that never works
    filled, foot, stats = lk.fill_gaps(dk, min_coverage=0.7, max_gap=6.0)
    assert not foot[5, 6] and foot[0, 0] and np.isnan(filled.u.values[:, 5, 6]).all()
    assert np.isfinite(filled.u.values[:, foot]).all() and stats["raw"] < stats["time"] <= stats["space"] <= 1.0
    assert abs(filled.u.values[11, 2, 3] - 0.5 * (dk.u.values[9, 2, 3] + dk.u.values[13, 2, 3])) < 1e-9


def test_godin_filter_removes_tides():
    import xarray as xr
    t = np.arange(24 * 20.0)
    sig = 0.3 + 0.5 * np.cos(2 * np.pi * t / 23.93) + 0.4 * np.cos(2 * np.pi * t / 12.42) + 0.01 * t / 24
    out = lk.godin_filter(xr.DataArray(sig, dims="time")).values
    assert np.isnan(out[:35]).all() and np.isfinite(out[36:-36]).all()
    np.testing.assert_allclose(out[40:-40], (0.3 + 0.01 * t / 24)[40:-40], atol=5e-3)


def test_ow_eddies_and_advect_curve():
    x = y = np.linspace(-50, 50, 101)
    X, Y = np.meshgrid(x, y)
    g = np.exp(-(X ** 2 + Y ** 2) / (2 * 10.0 ** 2))
    W, level, eddies, n = lk.ow_eddies(x, y, -Y * g, X * g)            # one Gaussian vortex
    assert len(eddies) == 1 and eddies[0]["cyclonic"] and abs(eddies[0]["xc"]) < 2 and 5 < eddies[0]["r_eq"] < 20
    c = lk.advect_curve(eddies[0]["boundary"], 0.0, 5.0, 0.05, solid_body(0.4))
    assert c["ratio"] == pytest.approx(1.0, abs=1e-3) and c["stay"] == 1.0


# -- second tracking backend ------------------------------------------------
def test_parcels_matches_rk4(bj):
    pytest.importorskip("parcels")
    x = np.linspace(0, bj.Lx, 201)
    y = np.linspace(-3000, 3000, 61)
    ds = lk.sample_to_dataset(bj, x, y, np.linspace(0, 10, 41))
    x0, y0 = np.array([4000.0, 8000.0, 2500.0, 100.0]), np.array([880.0, 0.0, 1300.0, 2900.0])
    vel = lk.grid_interpolator(ds, "u", "v")
    for t0, T in ((0.0, 5.0), (10.0, -5.0)):
        Xa, Ya, ta, tau_a = lk.advect(x0, y0, t0, T, 0.05, vel, save_every=20, return_exit=True)
        Xb, Yb, tb, tau_b = lk.advect_parcels(x0, y0, t0, T, 0.05, ds, save_every=20, return_exit=True)
        np.testing.assert_allclose(tb, ta)
        np.testing.assert_allclose(tau_b, tau_a, atol=0.05 + 1e-9)     # within one step
        np.testing.assert_allclose(Xb[:, :3], Xa[:, :3], atol=1e-3)    # km
        np.testing.assert_allclose(Yb[:, :3], Ya[:, :3], atol=1e-3)
        assert tau_a[3] < abs(T) and tau_b[3] < abs(T)                 # 4th one exits
