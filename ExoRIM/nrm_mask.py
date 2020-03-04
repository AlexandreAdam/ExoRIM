import poppy
import numpy as np
import astropy.units as u


def _r(x, y):
    return np.sqrt(x**2 + y**2)


class NRMask(poppy.AnalyticOpticalElement):
    """
    Defines a mask of N circular aperture of radius R 
    placed accros the wavefront in a non redundant manner
    """

    def __init__(self, name=None, radius=1.0 * u.meter, n=2, **kwargs):
        if name is None:
            name = f"NRM, N={N}, radius={radius}"
        super(NRMask, self).__init__(name=name, **kwargs)
        self.radius = radius
        self.n_aperture = n

    def get_transmission(self, wave):
        """
        Compute the transmission inside/outside the apertures
        """
        x, y = self.get_coordinates(wave)
        radius = self.radius.to(u.meter).value
        r = _r(x, y)
        del x
        del y


