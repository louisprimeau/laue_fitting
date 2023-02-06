from numpy import dot, cross, pi, sin, cos, tan, outer
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt
import gemmi
import imageio
import scipy.ndimage
from skimage.feature import blob_dog, blob_doh, blob_log
import skimage
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

# Blob detector
def blob_detection(image):
    template_size = 10
    template = np.zeros((template_size, template_size))
    x, y = np.meshgrid(np.linspace(-1, 1, template_size), np.linspace(-1, 1, template_size))
    r = np.sqrt(x**2 + y**2)
    template = np.exp(-(r/a)**2)
    template = (template - np.mean(template)) / np.std(template)
    peaks = scipy.ndimage.convolve(image, template)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[1].imshow(peaks)
    plt.show()

def sharpen_image(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) # ensure image is normalized
    #image = scipy.ndimage.gaussian_filter(image, sigma=4) # gausisan filter
    #mean, std = np.mean(image), np.std(image)
    #image = blob_dog(image, max_sigma=30, threshold=.1)
    #image[image < mean - 2*std] = mean - 2 * std # remove bottom 10%
    #image[image > mean - 2*std] = 0
    #image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image

def lowpass_filter(image):
    fimage = np.fft.fft2(image, norm='ortho')
    #x, y = np.meshgrid(np.linspace(-1, 1, image.shape[0]), np.linspace(-1, 1, image.shape[1]))
    #r = np.sqrt(x**2 + y**2)
    fig, ax = plt.subplots(1)
    ax.imshow(fimage.real)
    print(fimage)
    plt.show()
    #fimage[r < 0.8] = 0
    #image = np.fft.ifft2(fimage, norm='ortho').real
    #image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image

# Forward laue backscattering model
def forward_LAUE(a1, a2, a3, gemmi_small_molecule, hkl, max_hkl, phi, theta, gamma):

    # calculate reciprocal lattice vectors
    vol = a1.dot(cross(a2, a3))
    b1 = 2 * pi * cross(a2, a3) / vol
    b2 = 2 * pi * cross(a3, a1) / vol
    b3 = 2 * pi * cross(a1, a2) / vol

    # Make the crystal's rotation matrix w.r.t. the lab frame
    crystal_rotation_matrix = Rz(gamma) @ Ry(phi) @ Rz(theta)

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
    #n_keep = 250
    #idx = np.argsort(I)
    #I = I[idx][-n_keep:]
    #film_vector = film_vector[:, idx][:, -n_keep:]
    
    return film_vector, I

# crystal lattice vectors
#a1 = np.array([11.93, 0, 0]) # in A
#a2 = np.array([0, 14.30, 0]) # in A
#a3 = np.array([0, 0, 6.17]) # in A

# Corundrum
a,b,c = 4.754, 4.754, 12.99
alpha, beta, gamma = pi/2, pi/2, 2*pi/3
a1 = np.array([1, 0, 0]) * a
a2 = (Rz(alpha - pi/2) @ np.array([0, 1, 0])) * b
a3 = (Ry(gamma - pi/2) @ np.array([0, 0, 1])) * c

"""
# Quartz
a,b,c = 4.705, 4.705, 5.25
alpha, beta, gamma = pi/2, pi/2, 2*pi/3
a1 = np.array([1, 0, 0]) * a
a2 = (Rz(alpha - pi/2) @ np.array([0, 1, 0])) * b
a3 = (Ry(gamma - pi/2) @ np.array([0, 0, 1])) * c
"""
"""
# Silicon
a, b, c = 5.430941, 5.430941, 5.430941
a1 = np.array([1, 0, 0]) * a
a2 = np.array([0, 1, 0]) * b
a3 = np.array([0, 0, 1]) * c
"""

# crystal orientation
phi = 0 * pi / 180
theta = 0 * pi / 180
gamma = 43 * pi / 180

# beam axis
source_vector = np.array([0, 0, 1])

# max HKL due to limited wavelength
max_hkl = (2 * norm(a1) / 0.3)**2 # h^2 + k^2 + l^2 < (2a/l)^2 where a is lattice spacing and l is min wavelength of source
#max_hkl = 100

# max HKL to iterate over
hkl = 20

# distance of film
d = 30 # mm
xlim = 127 / 2 # 127 mm detector in x direction
ylim = 127 / 2 # 127 mm detector in y direction

# cif file structure factor calculator
small = gemmi.read_small_structure('corundrum.cif')
small.change_occupancies_to_crystallographic()

# forward model
spots, intensities = forward_LAUE(a1, a2, a3, small, hkl, max_hkl, phi, theta, gamma)

# plot with scatter Bailey style
fig, ax = plt.subplots(1)

image = np.asarray(imageio.imread('data/23_01_17_0001_ruby_01_0.bmp'))
image = np.mean(image, axis=2)
#image = lowpass_filter(image)
#image = sharpen_image(image)
#xlim, ylim = image.shape[0], image.shape[1]
#xcenter, ycenter = 513, 544
yshift = 0 #ycenter - image.shape[1]/2
xshift = -70 # xcenter - image.shape[0]/2

blobs = np.genfromtxt('data/23_01_17_0001_ruby_01_0.csv', delimiter=',')
blobs[:, 0] = (blobs[:, 0] + 0.5) * image.shape[0]
blobs[:, 1] = (blobs[:, 1] + 0.5) * image.shape[1]

spots[0, :] = xshift + (spots[0, :] / (2 * xlim) + 0.5) * image.shape[0]
spots[1, :] = yshift + (spots[1, :] / (2 * ylim) + 0.5) * image.shape[1]

ax.imshow(image, cmap='gray', vmin=0, vmax=255)
ax.scatter(spots[0, :], spots[1, :], s=5*intensities, alpha=0.2, label='Calculated Laue Spots')
ax.scatter(blobs[:, 0], blobs[:, 1], s=5, marker='x', color='k', label='Labelled Laue Spots')

ax.axvline(image.shape[0]/2 + xshift, 0, image.shape[0], linewidth=0.4, color='k')
ax.axhline(image.shape[1]/2 + yshift, 0, image.shape[0], linewidth=0.4, color='k')
ax.axvline(image.shape[0]/2, 0, image.shape[0], linewidth=0.1, color='b')
ax.axhline(image.shape[1]/2, 0, image.shape[0], linewidth=0.1, color='b')



phis = np.linspace(-np.arctan(ylim / d) / 2, np.arctan(ylim / d) / 2, 11)
yticks = yshift + (d * tan(2 * phis) / (2 * ylim) + 0.5) * image.shape[0]
ax.set_yticks(ticks=yticks, labels=["{:.2f}".format(label) for label in (phis + phi) * 180 / pi])
ax.set_ylabel('Phi')
gammas = np.linspace(-np.arctan(xlim / d) / 2, np.arctan(xlim / d) / 2, 11)
xticks = xshift + (d * tan(2 * gammas) / (2 * xlim) + 0.5) * image.shape[1]
ax.set_xticks(ticks=xticks, labels=["{:.1f}".format(label) for label in (gammas + gamma) * 180 / pi])
ax.set_xlabel('Gamma')
ax.legend(loc='lower right')


plt.show()
