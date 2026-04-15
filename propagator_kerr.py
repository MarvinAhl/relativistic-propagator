import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker

# The dynamics for a rotation black hole in spherical coordinates and natural units
def geodesic_dynamics(xdx, GM, a):
    r = xdx[1]
    theta = xdx[2]
    tdot = xdx[4]
    rdot = xdx[5]
    thetadot = xdx[6]
    phidot = xdx[7]

    # The geodesic equation (from symbolic notebook)
    ddx = np.array([2*GM*(-a**2*r*tdot*thetadot*np.sin(2*theta) + a*phidot*r*thetadot*(a**2 + r**2)*np.sin(2*theta) - a*phidot*rdot*(-a**2*np.cos(theta)**2 + r**2)*np.sin(theta)**2 + rdot*tdot*(-a**2*np.cos(theta)**2 + r**2))/((a**2*np.cos(theta)**2 + r**2)*(2*GM*r - a**2*np.cos(theta)**2 - r**2)), (rdot**2*(a**2*np.cos(theta)**2 + r**2)**2*(-r*(-2*GM*r + a**2 + r**2) + (-GM + r)*(a**2*np.cos(theta)**2 + r**2)) + thetadot*(a**2*np.cos(theta)**2 + r**2)**2*(a**2*rdot*np.sin(2*theta) + r*thetadot*(-2*GM*r + a**2 + r**2))*(-2*GM*r + a**2 + r**2) + (-2*GM*r + a**2 + r**2)**2*(2*GM*a*phidot*tdot*(-a**2*np.cos(theta)**2 + r**2)*np.sin(theta)**2 - GM*tdot**2*(-a**2*np.cos(theta)**2 + r**2) + phidot**2*(r*(a**2*(-2*GM*r + a**2 + r**2)*np.sin(theta)**2 - (a**2 + r**2)**2) + (a**2*np.cos(theta)**2 + r**2)*(a**2*(GM - r)*np.sin(theta)**2 + 2*r*(a**2 + r**2)))*np.sin(theta)**2))/((a**2*np.cos(theta)**2 + r**2)**3*(-2*GM*r + a**2 + r**2)), (GM*a**2*r*tdot**2*(-2*GM*r + a**2 + r**2)*np.sin(2*theta) - 1/8*a**2*rdot**2*(a**2*np.cos(2*theta) + a**2 + 2*r**2)**2*np.sin(2*theta) + phidot*(-4*GM*a*r*tdot*(a**2 + r**2) + phidot*(-a**2*(a**2*(-2*GM*r + a**2 + r**2)*np.sin(theta)**2 - (a**2 + r**2)**2)*np.sin(theta)**2 + (a**2*np.cos(theta)**2 + r**2)*(-2*a**2*(-2*GM*r + a**2 + r**2)*np.sin(theta)**2 + (a**2 + r**2)**2)))*(-2*GM*r + a**2 + r**2)*np.sin(theta)*np.cos(theta) + thetadot*(a**2*np.cos(theta)**2 + r**2)**2*((1/2)*a**2*thetadot*np.sin(2*theta) - 2*r*rdot)*(-2*GM*r + a**2 + r**2))/((a**2*np.cos(theta)**2 + r**2)**3*(-2*GM*r + a**2 + r**2)), 2*(-2*GM*a*r*tdot*thetadot*(a**2 + r**2)*np.cos(theta) + GM*a*rdot*tdot*(-a**2*np.cos(theta)**2 + r**2)*np.sin(theta) + phidot*rdot*(r*(a**2*(-2*GM*r + a**2 + r**2)*np.sin(theta)**2 - (a**2 + r**2)**2) + (a**2*np.cos(theta)**2 + r**2)*(a**2*(GM - r)*np.sin(theta)**2 + 2*r*(a**2 + r**2)))*np.sin(theta) - phidot*thetadot*(a**2*(a**2*(-2*GM*r + a**2 + r**2)*np.sin(theta)**2 - (a**2 + r**2)**2)*np.sin(theta)**2 + (a**2*np.cos(theta)**2 + r**2)*(2*a**2*(-2*GM*r + a**2 + r**2)*np.sin(theta)**2 - (a**2 + r**2)**2))*np.cos(theta))/((a**2*np.cos(theta)**2 + r**2)*(a**2*(-2*GM*r + a**2 + r**2)*np.sin(theta)**2 - (a**2 + r**2)**2)*np.sin(theta))])
    
    return np.hstack((xdx[4:8].copy(), ddx))

# Convert a state vector (four-position and four-velocity) from spherical natural to cartesian SI
def sphnat2cartsi(xdx_sph, c):
    t = xdx_sph[0]
    r = xdx_sph[1]
    theta = xdx_sph[2]
    phi = xdx_sph[3]
    tdot = xdx_sph[4]
    rdot = xdx_sph[5]
    thetadot = xdx_sph[6]
    phidot = xdx_sph[7]

    # Convert state from spherical to cartesian coordinates
    x_cart = np.array([
        t,
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta)
    ])

    # Converts spherical four-velocity to cartesian four-velocity (from symbolic notebook)
    dx_cart = np.array([tdot, -phidot*r*np.sin(phi)*np.sin(theta) + r*thetadot*np.cos(phi)*np.cos(theta) + rdot*np.sin(theta)*np.cos(phi), phidot*r*np.sin(theta)*np.cos(phi) + r*thetadot*np.sin(phi)*np.cos(theta) + rdot*np.sin(phi)*np.sin(theta), -r*thetadot*np.sin(theta) + rdot*np.cos(theta)])

    xdx_cart = np.hstack((x_cart, dx_cart))

    # Convert to SI units
    nat2si = np.array([1/c, 1, 1, 1, 1, c, c, c])

    xdx_cart_si = xdx_cart * nat2si

    return xdx_cart_si

# Convert a state vector (four-position and four-velocity) from cartesian SI to spherical natural
def cartsi2sphnat(xdx_cart_si, c):
    # Convert to natural units
    si2nat = np.array([c, 1, 1, 1, 1, 1/c, 1/c, 1/c])

    xdx_cart = xdx_cart_si * si2nat

    t_cart = xdx_cart[0]
    x_cart = xdx_cart[1]
    y_cart = xdx_cart[2]
    z_cart = xdx_cart[3]
    tdot_cart = xdx_cart[4]
    xdot_cart = xdx_cart[5]
    ydot_cart = xdx_cart[6]
    zdot_cart = xdx_cart[7]

    t = t_cart
    r = np.sqrt(x_cart**2 + y_cart**2 + z_cart**2)
    theta = np.acos(z_cart / r)
    phi = np.sign(y_cart) * np.acos(x_cart / (x_cart**2 + y_cart**2))

    # Convert the state
    x_sph = np.array([t, r, theta, phi])
    
    # Converts cartesian four-velocity to spherical four-velocitys (from symbolic notebook)
    dx_sph = np.array([tdot_cart, xdot_cart*np.sin(theta)*np.cos(phi) + ydot_cart*np.sin(phi)*np.sin(theta) + zdot_cart*np.cos(theta), (xdot_cart*np.cos(phi)*np.cos(theta) + ydot_cart*np.sin(phi)*np.cos(theta) - zdot_cart*np.sin(theta))/r, (-xdot_cart*np.sin(phi) + ydot_cart*np.cos(phi))/(r*np.sin(theta))])

    return np.hstack((x_sph, dx_sph))

# Returns tdot to normalize four-velocity
def norm4vel(xdx, GM, a):
    r = xdx[1]
    theta = xdx[2]
    rdot = xdx[5]
    thetadot = xdx[6]
    phidot = xdx[7]

    # tdot that satisfies the four-velocity norm constraint |U| = -1 (from symbolics notebook)
    return (-2*GM*a*phidot*r*(4*GM**2*r**2 - 2*GM*a**2*r*np.cos(theta)**2 - 2*GM*a**2*r - 4*GM*r**3 + a**4*np.cos(theta)**2 + a**2*r**2*np.cos(theta)**2 + a**2*r**2 + r**4)*np.sin(theta)**2 + np.sqrt((-2*GM*r + a**2 + r**2)*(4*GM**2*a**4*phidot**2*r**2*np.sin(theta)**6 - 8*GM**2*a**4*phidot**2*r**2*np.sin(theta)**4 + 4*GM**2*a**4*phidot**2*r**2*np.sin(theta)**2 + 4*GM**2*a**4*r**2*thetadot**2*np.sin(theta)**4 - 8*GM**2*a**4*r**2*thetadot**2*np.sin(theta)**2 + 4*GM**2*a**4*r**2*thetadot**2 - 8*GM**2*a**2*phidot**2*r**4*np.sin(theta)**4 + 8*GM**2*a**2*phidot**2*r**4*np.sin(theta)**2 - 8*GM**2*a**2*r**4*thetadot**2*np.sin(theta)**2 + 8*GM**2*a**2*r**4*thetadot**2 - 4*GM**2*a**2*r**2*np.sin(theta)**2 + 4*GM**2*a**2*r**2 + 4*GM**2*phidot**2*r**6*np.sin(theta)**2 + 4*GM**2*r**6*thetadot**2 + 4*GM**2*r**4 - 4*GM*a**6*phidot**2*r*np.sin(theta)**6 + 8*GM*a**6*phidot**2*r*np.sin(theta)**4 - 4*GM*a**6*phidot**2*r*np.sin(theta)**2 - 2*GM*a**6*r*thetadot**2*np.sin(theta)**4 + 4*GM*a**6*r*thetadot**2*np.sin(theta)**2 - 2*GM*a**6*r*thetadot**2*np.cos(theta)**6 - 2*GM*a**6*r*thetadot**2 - 4*GM*a**4*phidot**2*r**3*np.sin(theta)**6 + 16*GM*a**4*phidot**2*r**3*np.sin(theta)**4 - 12*GM*a**4*phidot**2*r**3*np.sin(theta)**2 - 8*GM*a**4*r**3*thetadot**2*np.sin(theta)**4 + 20*GM*a**4*r**3*thetadot**2*np.sin(theta)**2 - 12*GM*a**4*r**3*thetadot**2 - 2*GM*a**4*r*rdot**2*np.sin(theta)**4 + 4*GM*a**4*r*rdot**2*np.sin(theta)**2 - 2*GM*a**4*r*rdot**2 - 2*GM*a**4*r*np.sin(theta)**4 + 6*GM*a**4*r*np.sin(theta)**2 - 4*GM*a**4*r + 8*GM*a**2*phidot**2*r**5*np.sin(theta)**4 - 12*GM*a**2*phidot**2*r**5*np.sin(theta)**2 + 10*GM*a**2*r**5*thetadot**2*np.sin(theta)**2 - 12*GM*a**2*r**5*thetadot**2 + 4*GM*a**2*r**3*rdot**2*np.sin(theta)**2 - 4*GM*a**2*r**3*rdot**2 + 6*GM*a**2*r**3*np.sin(theta)**2 - 8*GM*a**2*r**3 - 4*GM*phidot**2*r**7*np.sin(theta)**2 - 4*GM*r**7*thetadot**2 - 2*GM*r**5*rdot**2 - 4*GM*r**5 + a**8*phidot**2*np.sin(theta)**6 - 2*a**8*phidot**2*np.sin(theta)**4 + a**8*phidot**2*np.sin(theta)**2 + a**8*thetadot**2*np.cos(theta)**6 + 2*a**6*phidot**2*r**2*np.sin(theta)**6 - 6*a**6*phidot**2*r**2*np.sin(theta)**4 + 4*a**6*phidot**2*r**2*np.sin(theta)**2 + 3*a**6*r**2*thetadot**2*np.sin(theta)**4 - 6*a**6*r**2*thetadot**2*np.sin(theta)**2 + a**6*r**2*thetadot**2*np.cos(theta)**6 + 3*a**6*r**2*thetadot**2 + a**6*rdot**2*np.cos(theta)**6 + a**6*np.sin(theta)**4 - 2*a**6*np.sin(theta)**2 + a**6 + a**4*phidot**2*r**4*np.sin(theta)**6 - 6*a**4*phidot**2*r**4*np.sin(theta)**4 + 6*a**4*phidot**2*r**4*np.sin(theta)**2 + 3*a**4*r**4*thetadot**2*np.sin(theta)**4 - 9*a**4*r**4*thetadot**2*np.sin(theta)**2 + 6*a**4*r**4*thetadot**2 + 3*a**4*r**2*rdot**2*np.sin(theta)**4 - 6*a**4*r**2*rdot**2*np.sin(theta)**2 + 3*a**4*r**2*rdot**2 + a**4*r**2*np.sin(theta)**4 - 4*a**4*r**2*np.sin(theta)**2 + 3*a**4*r**2 - 2*a**2*phidot**2*r**6*np.sin(theta)**4 + 4*a**2*phidot**2*r**6*np.sin(theta)**2 - 3*a**2*r**6*thetadot**2*np.sin(theta)**2 + 4*a**2*r**6*thetadot**2 - 3*a**2*r**4*rdot**2*np.sin(theta)**2 + 3*a**2*r**4*rdot**2 - 2*a**2*r**4*np.sin(theta)**2 + 3*a**2*r**4 + phidot**2*r**8*np.sin(theta)**2 + r**8*thetadot**2 + r**6*rdot**2 + r**6))*(-2*GM*r + a**2*np.cos(theta)**2 + r**2))/((-2*GM*r + a**2*np.cos(theta)**2 + r**2)*(4*GM**2*r**2 - 2*GM*a**2*r*np.cos(theta)**2 - 2*GM*a**2*r - 4*GM*r**3 + a**4*np.cos(theta)**2 + a**2*r**2*np.cos(theta)**2 + a**2*r**2 + r**4))


if __name__ == '__main__':
    # SI units
    C = 299792.458  # km / s, speed of light
    GM_SUN_SI = 1.32712440042e11  # km^3 / s^2, standard gravitational parameter of sun (just for reference)
    GM_SI = GM_SUN_SI * 1e5  # Super massive black hole
    A_SI = 1.0 * GM_SI / C  # km^2 / s, J/M spin parameter, factor 1.0 is a maximally spinning black hole
    # Factor 10.3 for spongebob flower orbit shapes

    # Natural units (c=1)
    GM = GM_SI / C**2  # km
    A = A_SI / C  # km

    # Event horizon
    Rs_si = 2 * GM_SI / C**2

    # Setup initial state
    init_dist = 0.8e6  # km
    init_vel = np.sqrt(GM_SI / init_dist) * 1.5  # km/s
    prop_time = 200  # s
    # 168.5 for spongebob flower orbit shapes

    # Define initial conditions
    # (t_dot doesn't matter as it's normalized in the next step)
    x0_cart_si = np.array([0, init_dist, 0, 0, 1, 0, 0, init_vel])
    x0 = cartsi2sphnat(x0_cart_si, C)

    # Normalize tdot such that four velocity is consistent: g * U * U = -1
    tdot_norm = norm4vel(x0, GM, A)
    x0[4] = tdot_norm

    tau0 = 0
    tau1 = prop_time * C  # Propagation time in seconds
    tau_span = np.linspace(tau0, tau1, 1000)

    # Event horizon crossing event
    crossing_event = lambda _, xdx: xdx[1] - Rs_si * 1.01  # Set minimum radius here
    crossing_event.terminal = True
    crossing_event.direction = -1
    
    sol = solve_ivp(lambda _, xdx: geodesic_dynamics(xdx, GM, A), t_span=(tau0, tau1), y0=x0,
                    method='DOP853', t_eval=tau_span, rtol=1e-13, atol=1e-13, events=crossing_event)
    taus = sol.t
    xdxs = sol.y.T

    # Check for event horizon crossings
    if sol.t_events[0].size > 0:
        print("Minimum radius crossing detected. Propagation terminated after tau = {} s.".format(taus[-1]/C))

    # Convert result to cartesian SI units
    taus_si = taus / C
    xdxs_cart_si = np.empty_like(xdxs)
    for i, xdx in enumerate(xdxs):
        xdxs_cart_si[i] = sphnat2cartsi(xdx, C)
    
    # Print time dilation
    print("Proper time: {} s".format(prop_time))
    print("Observer time: {} s".format(xdxs_cart_si[-1, 0]))
    print("Time dilation: {} s".format(xdxs_cart_si[-1, 0] - prop_time))

    # Plot
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')

    # Plot black hole event horizon sphere
    u_sph = np.linspace(0, 2 * np.pi, 30)
    v_sph = np.linspace(0, np.pi, 20)
    x_sph = Rs_si * np.outer(np.cos(u_sph), np.sin(v_sph))
    y_sph = Rs_si * np.outer(np.sin(u_sph), np.sin(v_sph))
    z_sph = Rs_si * np.outer(np.ones(np.size(u_sph)), np.cos(v_sph))

    ax.plot_surface(x_sph, y_sph, z_sph, color='#00000020')

    # Plot trajectory
    ax.plot(xdxs_cart_si[:, 1], xdxs_cart_si[:, 2], xdxs_cart_si[:, 3], 'k-')

    ax.axis('equal')
    ax.set_xlabel('x / km')
    ax.set_ylabel('y / km')
    ax.set_zlabel('z / km')

    formatter = ticker.ScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits((5, 5))  # force 10^5 scaling

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.zaxis.set_major_formatter(formatter)

    plt.show()