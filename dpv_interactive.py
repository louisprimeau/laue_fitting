"""
LAUE Orientation GUI
Louis Primeau
February 23rd, 2023
MIT Licence

This file creates a matplotlib widget GUI to manually fit Laue backscattering diagrams.
It ingests lattice parameters, angles, and a .cif file to calculate the Laue spots.
It ingests .bmp images and .csv labels to plot. The .csv files can be created using the accompanying labelling tool.
It gives the user the ability to modify the crystal Euler angles and distance to the detector in the GUI. 

Requirements:

Python 3.9.13
matplotlib 3.5.1
numpy 1.21.5
gemmi 0.5.7
imageio 2.19.3

"""
from numpy import dot, cross, pi, sin, cos, tan, outer
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt
import gemmi
import imageio.v2 as imageio
from matplotlib.widgets import Slider, Button

"""
FUNCTIONS
"""

# Rotation about x axis
def Rx(phi):
    c, s = cos(phi), sin(phi)
    return np.array([[1, 0, 0],
                     [0, c,-s],
                     [0, s, c]])

# Rotation about y axis
def Ry(phi):
    c, s = cos(phi), sin(phi)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

# Rotation about z axis
def Rz(phi):
    c, s = cos(phi), sin(phi)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0,  1]])

# reflect x across y
def reflect_across(x, y):
    return 2 * np.outer(x, (x @ y)) - y

# normalize array along an axis
def normalize(v, axis=0):
    return v / norm(v, axis=axis).reshape(*v.shape[:axis], 1, *v.shape[axis+1:])

# call gemmi structure factor calculator
def structure_factor(small_molecule, H, K, L):
    calc_x = gemmi.StructureFactorCalculatorX(small_molecule.cell)
    S = []
    for h, k, l in zip(H, K, L):
        S.append(calc_x.calculate_sf_from_small_structure(small, (h,k,l)))
    return np.abs(np.array(S)).real

# Preuss scattering intensity
def scattering_intensity(sigma):
    theta = pi/2 - sigma
    #theta[theta == 0] = 0.01
    return (1 - cos(2*theta)**2) * (1 - 1 / tan(theta)**2) * cos(2*sigma)**3 / sin(theta)**2

# Forward laue backscattering model
def forward_LAUE(a1, a2, a3, d, gemmi_small_molecule, hkl, max_hkl, phi, theta, gamma):

    # calculate reciprocal lattice vectors
    vol = a1.dot(cross(a2, a3))
    b1 = 2 * pi * cross(a2, a3) / vol
    b2 = 2 * pi * cross(a3, a1) / vol
    b3 = 2 * pi * cross(a1, a2) / vol

    # Make the crystal's rotation matrix w.r.t. the lab frame
    crystal_rotation_matrix = Rz(gamma) @ Ry(phi) @ Rx(theta)

    # Generate all possible hkl triplets
    H, K, L = np.meshgrid(np.arange(-hkl, hkl), np.arange(-hkl, hkl), np.arange(-hkl, hkl))
    H = H.reshape(-1); K = K.reshape(-1); L = L.reshape(-1)

    # get rid of spurious hkls
    mask1 = ((H == 0) & (K == 0)) & (L == 0)
    mask2 = (H**2 + K**2 + L**2) > max_hkl
    mask = ~mask1 & ~mask2
    H = H[mask]; K = K[mask]; L = L[mask]

    # allowed wavevectors in the crystal frame
    K_CF = normalize(outer(b1, H) + outer(b2, K) + outer(b3, L), axis=0) # 3 x N

    # allowed wavevectors in the lab frame
    K_LF = normalize(crystal_rotation_matrix @ K_CF, axis=0)

    # Angle of allowed wavevector and the beam axis
    sigma = np.arccos(-source_vector @ K_LF)

    # Get rid of wavevectors outside of 45 degree cone, as the reflected vectors will not hit film
    mask = ~(sigma > (pi/4 - 0.01))
    H = H[mask]; K = K[mask]; L = L[mask]
    K_LF = K_LF[:, mask]; sigma = sigma[mask]

    # Calculate reflected wavevectors
    K_R = normalize(reflect_across(source_vector, K_LF), axis=0)

    # Calculate film coordinates from reflected wavevectors
    film_vector = normalize(K_R[:2, :], axis=0)* d * tan(2 * sigma)

    # Enforce detector/film boundaries
    mask = (film_vector[0] > -xlim) & (film_vector[0] < xlim) & (film_vector[1] > -ylim) & (film_vector[1] < ylim)
    film_vector = film_vector[:, mask]
    H =	H[mask]; K = K[mask]; L = L[mask]
    sigma = sigma[mask]

    # Calculate spot intensities
    I = structure_factor(gemmi_small_molecule, H, K, L) * scattering_intensity(sigma)

    # Remove small intensity spots
    n_keep = 500 # Modify to suit your needs
    idx = np.argsort(I)
    I = I[idx][-n_keep:]
    film_vector = film_vector[:, idx][:, -n_keep:]
    
    return film_vector, I



"""
PARAMETERS
"""



"""
# Corundrum
a,b,c = 4.754, 4.754, 12.99
alpha, beta, gamma = pi/2, pi/2, 2*pi/3
a1 = np.array([1, 0, 0]) * a
a2 = (Rz(alpha - pi/2) @ np.array([0, 1, 0])) * b
a3 = (Ry(gamma - pi/2) @ np.array([0, 0, 1])) * c
"""

"""
# Quartz
a,b,c = 4.914, 4.914, 5.406
alpha, beta, gamma = pi/2, pi/2, 2*pi/3
a1 = np.array([1, 0, 0]) * a
a2 = (Rz(alpha - pi/2) @ np.array([0, 1, 0])) * b
a3 = (Ry(gamma - pi/2) @ np.array([0, 0, 1])) * c
"""

"""
# Copper
a, b, c = 3.615, 3.615, 3.615
alpha = beta = gamma = pi/2
a1 = np.array([1, 0, 0]) * a
a2 = np.array([0, 1, 0]) * b
a3 = np.array([0, 0, 1]) * c
"""

"""
# Silicon
a, b, c = 5.430941, 5.430941, 5.430941
a1 = np.array([1, 0, 0]) * a
a2 = np.array([0, 1, 0]) * b
a3 = np.array([0, 0, 1]) * c

"""

"""
# Germanium
a, b, c = 5.65754,  5.65754, 5.65754
a1 = np.array([1, 0, 0]) * a
a2 = np.array([0, 1, 0]) * b
a3 = np.array([0, 0, 1]) * c
"""

# Apophyllite
a, b, c = 8.965, 8.965, 15.768
a1 = np.array([1, 0, 0]) * a
a2 = np.array([0, 1, 0]) * b
a3 = np.array([0, 0, 1]) * c

# initial crystal orientation
phi = 0 * pi / 180
theta = 0 * pi / 180
gamma = 0 * pi / 180

# beam axis
source_vector = np.array([0, 0, 1])

# max HKL due to limited wavelength
max_hkl = (2 * norm(a1) / 0.3)**2 # h^2 + k^2 + l^2 < (2a/l)^2 where a is lattice spacing and l is min wavelength of source

# max HKL to iterate over
hkl = 20

# distance of film
d = 30 # mm
xlim = 127 / 2 # 127 mm detector in x direction
ylim = 127 / 2 # 127 mm detector in y direction

# cif file structure factor calculator
small = gemmi.read_small_structure('apophyllite.cif')
small.change_occupancies_to_crystallographic()

# Read image from disk
image = np.asarray(imageio.imread('data/23_01_31_0001_apophyllite_0.bmp'))
image = np.mean(image, axis=2)

# Beam center offset in pixels
xshift = 0
yshift = 0



"""
PLOTTING
"""


# forward model
spots, intensities = forward_LAUE(a1, a2, a3, d, small, hkl, max_hkl, phi, theta, gamma)
spots[0, :] = xshift + (spots[0, :] / (2 * xlim) + 0.5) * image.shape[0]
spots[1, :] = yshift + (spots[1, :] / (2 * ylim) + 0.5) * image.shape[1]

# plot with scatter Bailey style
fig, ax = plt.subplots(1)
plt.subplots_adjust(bottom=0.40)

# Make sliders
axtheta = plt.axes([0.25, 0.25, 0.65, 0.03])
thetaS = Slider(axtheta, 'Theta', 0.0, 2 * pi, 0.0)

axphi = plt.axes([0.25, 0.20, 0.65, 0.03])
phiS = Slider(axphi, 'Phi', 0.0, pi, 0.0)

axgamma = plt.axes([0.25, 0.15, 0.65, 0.03])
gammaS = Slider(axgamma, 'Gamma', 0.0, 2 * pi, 0.0)

axd = plt.axes([0.25, 0.1, 0.65, 0.03])
dS = Slider(axd, 'd', 20, 40, 30)

# Read labelled data from disk
blobs = np.genfromtxt('data/23_01_31_0001_apophyllite_0.csv', delimiter=',')
blobs[:, 0] = (blobs[:, 0] + 0.5) * image.shape[0]
blobs[:, 1] = (blobs[:, 1] + 0.5) * image.shape[1]

# Populate plot
ax.imshow(image, cmap='gray', vmin=0, vmax=255)
old_scatter = ax.scatter(spots[0, :], spots[1, :], s=5*intensities, alpha=0.2, label='Calculated Laue Spots')
ax.scatter(blobs[:, 0], blobs[:, 1], s=5, marker='x', color='k', label='Labelled Laue Spots')

# Slider update function
def update(val):
    phi = phiS.val
    theta = thetaS.val
    gamma = gammaS.val
    d = dS.val
    
    spots, intensities = forward_LAUE(a1, a2, a3, d, small, hkl, max_hkl, phi, theta, gamma)
    spots[0, :] = xshift + (spots[0, :] / (2 * xlim) + 0.5) * image.shape[0]
    spots[1, :] = yshift + (spots[1, :] / (2 * ylim) + 0.5) * image.shape[1]

    xx = np.vstack ((spots[0, :], spots[1, :]))
    old_scatter.set_offsets (xx.T)
    old_scatter.set_sizes(intensities * 5)
    
    fig.canvas.draw_idle()
    
thetaS.on_changed(update)
phiS.on_changed(update)
gammaS.on_changed(update)
dS.on_changed(update)

# crosshairs
ax.axvline(image.shape[0]/2, 0, image.shape[0], linewidth=0.1, color='b')
ax.axhline(image.shape[1]/2, 0, image.shape[0], linewidth=0.1, color='b')

# axis angle labels
phis = np.linspace(-np.arctan(ylim / d) / 2, np.arctan(ylim / d) / 2, 11)
yticks = yshift + (d * tan(2 * phis) / (2 * ylim) + 0.5) * image.shape[0]
ax.set_yticks(ticks=yticks, labels=["{:.2f}".format(label) for label in (phis) * 180 / pi])
ax.set_ylabel('angle')
gammas = np.linspace(-np.arctan(xlim / d) / 2, np.arctan(xlim / d) / 2, 11)
xticks = xshift + (d * tan(2 * gammas) / (2 * xlim) + 0.5) * image.shape[1]
ax.set_xticks(ticks=xticks, labels=["{:.1f}".format(label) for label in (gammas) * 180 / pi])
ax.set_xlabel('angle')
ax.legend(loc='lower right')


plt.show()
