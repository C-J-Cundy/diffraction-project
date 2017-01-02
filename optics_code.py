import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class Mirror:
    """Class encapsulating a mirror. The class contains a complex 
    array representing the aperture function and another (complex)
    array may be generated representing the far-field diffraction
    pattern"""

    points_per_m = 10
    pad_factor = 20
    #Number of times larger the padded array sides are compared to 2R
    #e.g. pad_factor = 1 corresponds to no padding.
    
    
    def taper(self, x, y, sigma, side):
        """Given an x,y cell index for an array with side length side, return 
        the value of a 2-D Gaussian function centred on the centre of the
        array with standard deviation sigma at the point x,y. All values
        are in terms of cell indices."""
        return np.exp(-((x - side / 2.0) ** 2 +
                        (y - side / 2.0) ** 2)
                      / (2 * sigma ** 2))
    
    def __init__(self, R, sigma_T):
        """Initialise the class, generating the aperture function array
        as a circle of radius R, tapered with the supplied value of sigma.
        """
        self.sigma_T = sigma_T
        self.R = R
        self.L = R * self.points_per_m #Number of array cells in a radius
        self.side = 2 * self.pad_factor * self.L
        #Number of cells in the padded aperture array
        self.aperture = np.fromfunction(lambda i, j:
                                        self.taper(i, j,
                                                   self.sigma_T *
                                                   self.points_per_m,
                                                   2 * self.L),
                                        (2 * self.L,
                                         2 * self.L), dtype=complex)
        #Above line generates a square array where each cell has the
        #value obtained by calling taper on that x,y coordinate

        y,x = np.ogrid[-self.L : self.L,
                       -self.L : self.L]
        mask = x*x + y*y >= self.L ** 2
        self.aperture[mask] = 0
        #Set cells at radius > R to 0

        #Embed in padded array
        padded = np.zeros([self.side, self.side], dtype=complex)
        padded[self.side / 2 - self.L: self.side / 2 + self.L,
               self.side / 2 - self.L: self.side / 2 + self.L] = \
                                                                 self.aperture
        self.aperture = padded
        
    def pattern(self, wavelength):
        """Calculate the far-field diffraction pattern with an FFT"""
        self.pattern = np.fft.fft2(self.aperture)
        self.pattern = np.fft.fftshift(self.pattern)
        self.scaling_factor = float(wavelength) / (float(self.side) /
                                                   self.points_per_m)
        #Angular width that a pixel corresponds to in the pattern array
        #in radians

    def find_width(self, power):
        """Given a power, finds the width between the two points in the
        central row of the pattern where the points have the value closest
        to power. NB assumes symmetric pattern, returns width in degrees."""
        
        #First fix up any problems that could occur due to phase differences
        row = np.absolute(self.pattern[len(self.pattern) / 2])
        power = abs(power)
        index = np.argmin(np.absolute(row - power)) #find index of value
        return abs(index - len(self.pattern) / 2) * 2 * (self.scaling_factor
                                                         * 180 / math.pi)
        #Return physical angular width in degrees
    
    
    def add_hole(self, radius):
        """Adds a hole of radius r to the telescope, by setting the 
        aperture function to zero in this region."""
        y,x = np.ogrid[-self.side / 2:self.side / 2,
                       -self.side / 2:self.side / 2]
        mask = x*x + y*y <= (radius * self.points_per_m) ** 2
        self.aperture[mask] = 0

    def add_point_errors(self, sigma_eps, wavelength):
        """Multiplies every point in the aperture by a phase error
        4*pi*epsilon/wavelength, where each epsilon is gaussian-disistributed 
        with standard deviation sigma_eps"""
        epsilon_array = np.random.normal(0, sigma_eps, [(self.side),
                                                        (self.side)])
        phase_array = np.exp(1j * epsilon_array * 4 * np.pi / wavelength)
        self.aperture = self.aperture * phase_array
                        

    def add_large_deformations(self, l_c, max_amp, n, wavelength):
        """Adds n 2-D gaussian functions with characteristic length l_c 
        (= sigma) and uniformly distributed amplitude to the aperture"""
        #Cut the Gaussians off at 3 sigma
        for i in range(0,n):
            #Select random coordinates in the grid. We choose ones that are
            #in the rectangle that the aperture is inscribed in
            array_l_c = int(math.floor(self.points_per_m * l_c))
            x, y = np.floor(np.random.random(2) * 2 * self.L + 
                            (self.pad_factor - 1) * self.L)
            amplitude = np.random.random() * max_amp
            gauss_array = amplitude * np.fromfunction(lambda i, j:
                                                      self.taper(i, j,
                                                                 (array_l_c),
                                                                 6 * array_l_c),
                                                      (math.floor(6 * array_l_c),
                                                       math.floor(6 * array_l_c)),
                                                      dtype=complex)
        
            self.aperture[x - 3 * array_l_c: x + 3 * array_l_c,
                          y - 3 * array_l_c: y + 3 * array_l_c] = \
            (np.exp(4 * 1j * math.pi * gauss_array / wavelength) *
             self.aperture[x - 3 * array_l_c: x + 3 * array_l_c,
                           y - 3 * array_l_c: y + 3 * array_l_c])






            




#-------------------------------------------------------------------------------
#Example of plotting code - reproducing figure 4 in the report
plt.subplot(2,1,1)
taper = Mirror(6,3)
taper.pattern(0.001)
mid = len(taper.pattern) / 2
x = np.linspace( -100 * taper.scaling_factor * 180 / np.pi,
                  100 * taper.scaling_factor * 180 / np.pi,
                 200)
#x-axis in degrees

#Fixes plot showing odd whitespace
plt.xlim([x[0], x[-1]])
plt.ylim([x[0], x[-1]])

plt.title(r'$\sigma/R = 1/2$')
plt.ylabel(r'$\phi$ displacement \ deg')

#Take central 200 * 200 pixels
plt.pcolor(x, x, np.log(np.absolute(taper.pattern[mid - 100: mid + 100,
                      mid - 100: mid + 100])),
              cmap='Greys_r')
plt.colorbar()

plt.subplot(2,1,2)
taper = Mirror(6, 2) #Re-generate with sharper tapering
taper.pattern(0.001) 
plt.pcolor(x, x, np.log(np.absolute(taper.pattern[mid - 100: mid + 100,
                      mid - 100: mid + 100])),
              cmap='Greys_r')
plt.colorbar()
plt.ylabel(r'$\phi$ displacement \ deg')
plt.xlabel(r'$\theta$ displacement \ deg')
plt.title(r'$\sigma/R = 1/3$')

#Fixes plot showing odd whitespace
plt.xlim([x[0], x[-1]])
plt.ylim([x[0], x[-1]])


plt.show()
