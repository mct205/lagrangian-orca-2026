"""Discussion cells of notebook 02, written from the executed figures (Gulf of Tonkin, 10-20 January 2015)."""

DISC = {}

DISC["C2"] = r"""
- **Coverage.** The same 1090 cells are valid at every hour: the top row is what continuous, gap-filled data look like. [C4](#C4) shows what the same data look like with gaps.
- **Mean current.** A southward current follows the coast, up to 0.25 m/s, typical of the winter monsoon. Offshore the mean flow is weak.
- **Variability.** The eddy kinetic energy is largest in the south (more than 400 cm² s⁻²). Section C3 shows that most of it is tidal. Hourly speeds average 0.16 m/s and reach about 1 m/s.
- **Gradients are weak and smooth.** The vorticity is at most ±0.15 $f$ (rms 0.10 $f$), with structures tens of kilometres wide and nothing at the grid scale. This is the signature of the 6 km resolution and of the 2DVar interpolation. Lagrangian structures come from velocity *gradients*, so this panel already tells you to expect broad, weak features.
- **Numerical check.** The fastest particle moves 1.7 km per integration step, well below the 6 km grid spacing, so the time step is adequate ([A9](00-lagrangian-theory.ipynb#A9)).
"""

DISC["C3"] = r"""
- **The flow is dominated by the diurnal tide.** In the spectrum the two largest peaks are at K1 and O1, with a smaller one at M2. The diurnal band holds about three quarters of the variance of the footprint-mean velocity; no separate inertial peak stands out.
- **The Godin filter removes 85% of the velocity variance** in this window. What is left (thick lines) varies over several days: at the centre of the footprint a southward flow of 0.15 m/s that relaxes during the period.
- So the two fields are very different in amplitude. Whether they also differ in their Lagrangian structures is a separate question, answered in [C8](#C8).
- Ten days cannot resolve K1 from O1 (their beat period is 13.7 days), which is why a filter, and not a harmonic fit, is used on a short window.
"""

DISC["C4"] = r"""
- **A 16% loss of data is a total loss of trajectories.** With gaps, only 38% of the particles are still tracked after 6 hours, 5% after a day and none after three days. A particle is dropped the first time it meets a missing value, in space or in time, and the first outage of the whole radar (at hour 30) removes everything that was left. No statistic and no map can be computed from such a field.
- **After filling, particles are lost only by leaving the footprint.** The survival curve of the filled field follows that of the complete field (71% against 73% after three days).
- **How wrong is the filling?** Where data were removed, the filled velocity differs from the truth by 0.10 m/s rms, a large fraction of the rms speed there (0.23 m/s). Yet the trajectories differ by only 0.5 km after three days, because the errors are short and scattered and largely average out along a path.
- **This test flatters the simple method.** The truth here is a smooth 2DVar field with regular tides, which is easy to interpolate, and the gaps were placed at random. Real gaps are worst at long range and at night, last longer, and hide the small-scale structures that a neighbour average cannot rebuild. Exercise 4 makes the test harder.
- This is why the data were re-processed with 2DVar before any Lagrangian analysis, and why the choice of gap-filling method is part of the result.
"""

DISC["C5"] = r"""
- **Where the water goes.** Particles released near the coast are carried south by the coastal current and leave through the southern edge of the footprint; a quarter of all particles are gone after 72 hours. Offshore patches move a few tens of kilometres and stay.
- **Raw and residual trajectories end in nearly the same place** (net displacement 35 and 39 km). The tide adds small back-and-forth excursions along the way: the distance travelled along the path is only a few kilometres longer in the raw field. A tidal current of 0.2 m/s reversing every 12 hours moves a particle by a few kilometres and brings it back.
- **The integrator is verified on real data:** OceanParcels and `lagkit` agree to round-off after 24 hours.
- The exits are not a defect of the method; they are the limit set by the size of the footprint. Everything that follows has to account for them.
"""

DISC["C6"] = r"""
- **Survival.** 90% of the particles are still inside after one day, 75% after three, 69% after five. The particles that leave are those in the fast coastal current, so the survivors are a biased sample: the slow part of the flow. Dispersion computed from them is an underestimate.
- **Dispersion.** It grows roughly ballistically during the first day and more slowly afterwards; after 24 hours the cloud has an rms size of about 8 km (east–west) by 11 km (north–south).
- **The autocorrelation oscillates with a 24-hour period** and returns to 0.75 after one day: the Lagrangian velocity is mostly the diurnal tide, which a particle meets again every day. $R_u$ never reaches zero within the record and $R_v$ only after 57 hours.
- **So $T_L$ and $K$ are not meaningful here.** The "first zero crossing" rule gives about a day and diffusivities above 1000 m²/s, but that is the tide, not turbulent decorrelation. On the residual field (Exercise 1) the oscillation disappears and the numbers change completely.
- The record is also short: 120 hours is only five tidal cycles and a few integral time scales, for a statistic that assumes $t\gg T_L$ ([A2](00-lagrangian-theory.ipynb#A2)).
"""

DISC["C7"] = r"""
- **The FSLE now covers 2 to 64 km.** With pairs released at nine separations, every threshold up to 32 km is crossed by about 1000 pairs. The grey curve is what the 2 km pairs alone give: it stops at 8 km with 13 pairs.
- **A plateau: $\lambda\approx0.20$ day⁻¹ from 2 to 32 km.** Pairs 2 km apart and pairs 32 km apart separate at the same relative rate. This is the non-local regime of [A4](00-lagrangian-theory.ipynb#A4), as in the Bickley jet: separation at all these scales is driven by the same large-scale strain and shear, here that of the coastal current. The left panel says the same thing: the $\langle D^2\rangle$ curves are parallel on the semilog plot, with a growth rate of 0.11 day⁻¹ whatever the initial separation.
- **Survivorship, twice.** (1) Beyond the largest initial separation (32 km, dotted line) a bin only contains pairs that grew into it: few pairs, and the fastest ones, so $\lambda$ jumps up. Do not interpret those points. (2) Even on the plateau only about a third of the released pairs cross a given threshold within 120 hours; the FSLE is the rate of that faster third, which is why it is about twice the rate of $\langle D^2\rangle$. The grey curve shows the same bias in a stronger form: its values climb to 0.3 day⁻¹ because only the fastest 2 km pairs reach the next thresholds.
- **What the plateau does and does not say.** Below the 6 km grid spacing the velocity difference between two particles comes from linear interpolation, so a constant $\lambda$ there is built in. Above it the pairs feel resolved gradients, and the plateau is a property of this velocity product. The 2DVar field is smooth, so it cannot show the scale-dependent (local) dispersion that real drifters might reveal at 5–30 km.
- The kurtosis grows steadily to about 4.5 for the small separations: the pairs become more and more unequal, as expected when a minority separates exponentially.
"""

DISC["C8"] = r"""
- **Coverage of the map.** About 90% of the footprint keeps a value for $T$ = 48 h; the rest, along the southern and coastal edges, lost its particles too early and is blank.
- **Weak, broad structures.** Typical values are 0.1 day⁻¹ (median) and 0.2–0.4 day⁻¹ (90th percentile), with a maximum of 0.7 day⁻¹. The maps show patches tens of kilometres wide, not the thin ridges of the Bickley jet. The strongest forward stretching is where the coastal current accelerates and peels away from the slower water offshore; the strongest backward values lie along the coast, where water converges onto the current.
- **A few negative values** (1–3% of the points). They mean that a fluid element shrank in every direction, which cannot happen in a non-divergent flow. Surface currents are divergent ([A1](00-lagrangian-theory.ipynb#A1)).
- **The tide hardly changes the structures.** Raw and residual maps are almost identical (pattern correlation 0.99 forward, 0.95 backward), although the filter removed 85% of the velocity variance. The diurnal tide here is a large-scale oscillation: it moves neighbouring particles together and brings them back, so it produces little net stretching. The structures are set by the residual flow.
- This is a property of this place and this product, not a general rule: where the tidal current has strong horizontal gradients (straits, headlands, shallow banks) it does create Lagrangian structures.
"""

DISC["C9"] = r"""
- **The classical parameters are out of reach.** Not a single point separates by a factor of 4 in 96 hours, and only 7% by a factor of 2. The stretching is too weak and the footprint too small: about a quarter of the points leave before any threshold is reached.
- **Only the ratio matters.** Doubling $\delta_0$ at the same ratio gives the same result. Below the grid spacing the velocity is interpolated linearly, so separations grow by a factor that does not depend on their size.
- With a ratio of 1.5 the map can be drawn (28% of the points reach $\delta_f$), and it highlights the same regions as the backward FTLE. With such a small ratio it is essentially a thresholded FTLE and carries no information about finite scales.
- **Three kinds of points, three treatments:** a value where $\delta_f$ was reached, zero where the pair never separated, and *blank* where the particles left first. Setting the last kind to zero would paint the coastal current, the most active part of the flow, as a quiet region.
- An FSLE map earns its name when $\delta_f$ is several grid cells. That needs stronger stretching, a larger domain, or both.
"""

DISC["C10"] = r"""
- **No eddy.** The largest LAVD value corresponds to 0.18 turns in 72 hours. The extraction does find one convex contour (radius 32 km, in the north-west), but the water inside turns by 0.15 turns: it fails the minimum-rotation test.
- **Coherent, but not an eddy.** The advected boundary keeps its perimeter (×0.94) and stays inside the footprint. The patch is not torn apart, simply because the flow there is weak and smooth. In a weak flow almost any closed curve stays "coherent" for three days, which is why the advection test alone is not sufficient either.
- **Okubo–Weiss marks the same place:** a cyclonic region of similar size (36 km) in the mean flow, plus two regions cut by the edge of the footprint. It has no way of saying that the rotation is too slow to matter, and part of its boundary leaves the footprint.
- **The LAVD is defined on two thirds of the footprint only**: the rest left within 72 hours. The coastal current, where the vorticity is strongest, is exactly where no value can be computed.
- "No coherent eddy in this window" is a result. Tuning `CD_MAX`, `MIN_ROTATION` or the period until a contour appears would be fitting the method to the wish (Exercise 5).
"""

DISC["SUMMARY_ROWS"] = r"""| Trajectories | southward coastal current; a quarter of the particles leave in 3 days | size of the footprint |
| Single-particle statistics | dispersion of about 10 km in a day; autocorrelation oscillating at 24 h | the tide dominates; survivors are a biased sample; short record |
| Relative dispersion, FSLE statistic | plateau of 0.2 day⁻¹ from 2 to 32 km (non-local regime) | needs pairs released at many separations; survivors only; smooth field hides any scale dependence |
| FTLE | broad patches, 0.1–0.4 day⁻¹; the same with and without the tide | smooth gap-filled field; exits at the edges |
| FSLE map | possible only for a separation ratio of about 1.5 | weak stretching; then equivalent to a thresholded FTLE |
| LAVD | one convex contour, 0.15 turns: no eddy | weak rotation; one third of the footprint lost to exits |
| Okubo–Weiss | one closed cyclonic region at the same place | says nothing about rotation rate or material coherence |"""

DISC = {k: v.strip("\n") for k, v in DISC.items()}
