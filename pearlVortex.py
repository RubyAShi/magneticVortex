import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageDraw
from scipy.ndimage import convolve

pi = np.pi
# magnetic permearbility in H/m
mu_0 = 4 * pi * np.float32(1e-7)
# Bohr magneton in J/T
m_0 = 9.274 * np.float32(1e-24)
# flux quanta in um^2 * G
phi_0 = 20.7

# genetrates magnetic field of a single dipole
class PearlVortex():

    def __init__(self, Rx=10, Ry=10, delta=.1, z0=2, Lambda=500):
        # Rx x scanning range in um
        # Ry y scanning range in um
        # delta pixel size
        # z0 scanning height
        # x0, y0 center of Pearl vortex
        self.Rx = Rx
        self.Ry = Ry
        self.delta = delta
        self.z0 = z0
        self.Lambda = Lambda

    def getVortex(self):
        Rx = self.Rx
        Ry = self.Ry
        delta = self.delta
        z0 = self.z0
        Lambda = self.Lambda

        # mesh the plotting area
        x = np.arange(-Rx, Rx + delta, delta)
        y = np.arange(-Ry, Ry + delta, delta)
        kmax = pi/ delta
        kdel = 2 * pi/ (len(x) * delta)
        kx = np.arange(-kmax, kmax + kdel, kdel)
        ky = np.arange(-kmax, kmax + kdel, kdel)
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX ** 2 + KY ** 2)
        # calculate expression of Pearl vortex in Fourier space
        hzk = phi_0 * np.exp(-K * z0)/(1 + K * Lambda)
        # shift Forier expression
        hzk = np.fft.fftshift(hzk)
        # inverse shifted Fourier expression
        hz = np.abs(np.fft.ifft2(hzk))
        # get unit right for B field expression
        hz = np.fft.fftshift(hz)/delta ** 2
        # calculate total flux in simulated image
        a = hz.sum() * delta ** 2 / phi_0
        total_flux = np.round(hz.sum() * delta ** 2 / phi_0, 4)
        title_string1 = 'total flux is '+ str(total_flux) + '$\phi_0$'
        title_string2 = 'at height ' + str(z0) + 'um'

        fig, ax = plt.subplots(1, 2)
        fig.tight_layout()
        im1 = ax[0].imshow(np.real(hzk), cmap=cm.RdYlGn, origin='lower', extent=[np.min(kx), np.max(kx), np.min(ky), np.max(ky)])
        ax[0].set_title('Fourier expression')
        ax[0].set_xlabel('kx(1/um)', labelpad=-2)
        ax[0].set_ylabel('ky(1/um)', labelpad=-2)
        cbar1 = fig.colorbar(im1, ax=ax[0], shrink=0.48)
        cbar1.ax.set_title('$\phi_0$')
        im2 = ax[1].imshow(hz, cmap=plt.cm.RdYlGn, origin='lower', extent=[-Rx, Rx, -Ry, Ry])
        ax[1].set_title(title_string1 +'\n' + title_string2)
        ax[1].set_xlabel('um', labelpad=-2)
        ax[1].set_ylabel('um', labelpad=-10)
        cbar2 = fig.colorbar(im2, ax=ax[1], shrink=0.48)
        cbar2.ax.set_title('$G$')
        plt.show()
        return hz

class SquidLayout():

    def __init__(self, type = 'IBM3um', xRange=[0,0], yRange=[0,0],z = 2, delta = .1,
                 xCenter = 35, yCenter = 38, mask = None, scaledMask = None, phi = 4, sigma = 0):
        self.type = type
        self.xRange = xRange
        self.yRange = yRange
        self.z = z
        self.delta = delta
        self.xCenter = xCenter
        self.yCenter = yCenter
        self.mask = mask
        self.scaledMask = scaledMask
        self.phi = phi
        self.sigma = sigma

    def get_rect(self, x, y, width, height, angle):
        rect = np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
        theta = (np.pi / 180.0) * angle
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        offset = np.array([x, y])
        transformed_rect = np.dot(rect, R) + offset
        return transformed_rect

    # SQUID geometry of a 3um IBM SQUID
    # grid size is 0.1
    # outputs a matrix "mask" of size 69 * 69
    def IBM3um_mask(self):
        if self.type != 'IBM3um':
            print('wrong type')
        else:
            data = np.zeros(69 * 69).reshape(69, 69)

            # Convert the numpy array to an Image object.
            img = Image.fromarray(data)

            # Draw a rotated rectangle on the image.
            draw = ImageDraw.Draw(img)
            rect = self.get_rect(x=32, y=0, width=4, height=7, angle=0)
            draw.polygon([tuple(p) for p in rect], fill=1)
            draw.ellipse((4, 8, 64, 68), fill=1)
            # Convert the Image data to a numpy array.
            mask = np.asarray(img)

            # Display the result using matplotlib.
            # this plots from top to bottom
            # first row of matrix is the top



            fig, ax = plt.subplots(1, 2, num = 3)
            ax[0].imshow(mask, cmap=plt.cm.gray)
            ax[0].set_title('matrix plot')
            ax[1].imshow(mask, cmap=plt.cm.gray, extent=[-3.4, 3.4, -3, 3.8])
            ax[1].set_title('real unit plot')
            plt.xlabel("um", labelpad=-2)
            plt.ylabel("um", labelpad=-5)
            plt.show()

            self.type = 'IBM3um'
            self.xRange = [-3.4, 3.4]
            self.yRange = [-3, 3.8]
            self.delta = .1
            self.xCenter = 35
            self.yCenter = 38
            self.mask = mask

class FlatConvolve(PearlVortex, SquidLayout):

    def __init__(self, Rx=50, Ry=50, B_delta= None, z0=2, Lambda = 500,
                 type='IBM3um', xRange=[0,0], yRange=[0,0], z=None, delta=.1,
                 xCenter = 5, yCenter = 6, mask=None, scaledMask =None):
        #Rx=10, Ry=10, delta=.1, z0=2, Lambda=500
        PearlVortex.__init__(self, Rx,Ry,B_delta,z0,Lambda)
        SquidLayout.__init__(self, type, xRange, yRange, z, delta,
                 xCenter, yCenter, mask, scaledMask)
        self.z = self.z0
        self.B_delta = self.delta

    def ConvolveFlat(self):
        Rx = self.Rx
        Ry = self.Ry
        delta = self.delta
        mask = self.mask

        B_Pearl = self.getVortex()
        flux = convolve(B_Pearl, mask)
        Phi_0 = 20.7
        flux = flux * delta ** 2 / Phi_0 * 1000

        fig, ax = plt.subplots()
        im = ax.imshow(flux, cmap=plt.cm.YlGnBu,
                       origin='lower', extent=[-Rx,Rx,-Ry,Ry])

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('m$\Phi_0$', size=14)
        plt.xlabel('um', size=14)
        plt.ylabel('um', size=14)
        plt.title('magnetic flux')
        plt.show()

        return flux
