# Relativistic Propagator
Author: Marvin Ahlborn</br>
Date: 2026-04-15

This is a basic propagator for trajectories in highly relativistic environments, e.g. close to black holes or neutron stars. The equations of motion are derived in two Jupyter notebooks using sympy from the geodesic equation for the Schwarzschild metric (no rotation) and the Kerr metric (with rotation). The resulting IVP is solved numerically in two python scripts which also implement conversions between the different units and coordinate systems.

Example of a polar orbit around a maximally spinning black hole with 1e5 solar masses. In contrast to Keplerian dynamics and non-rotating black holes, the orbit is not confined to a fixed plane. The trajectory's proper time is 200 s with a time dilation of 64 s with respect to a distant observer.</br>
<img src="https://github.com/MarvinAhl/relativistic-propagator/blob/main/plots/max_spin_polar.png" alt="Polar orbit around rotating black hole." width="500"/>

Example of a fly-by at a non-rotating black hole. The apsidal precession is so strong that a loop around the black hole is performed.</br>
<img src="https://github.com/MarvinAhl/relativistic-propagator/blob/main/plots/schwarzschild_loop.png" alt="Fly-by at non-rotating black hole." width="500"/>

The first case but with impossibly large spin (~10 times the maximum spin rate) leads to interesting orbit patterns.</br>
<img src="https://github.com/MarvinAhl/relativistic-propagator/blob/main/plots/over_spin_polar.png" alt="Polar orbit around impossibly fast rotating black hole." width="500"/>
