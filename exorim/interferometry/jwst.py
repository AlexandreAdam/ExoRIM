import morphine
import webbpsf

#get WebbPSF NIRISS model
niriss = webbpsf.NIRISS()
niriss.filter = "F380M"
niriss.pupil_mask = "MASK_NRM"
optsys = niriss.get_optical_system()

morphine_niriss = morphine.OpticalSystem(name="ami_niriss")

# entrance pupil
morphine_niriss.add_pupil(morphine.ArrayOpticalElement(
    transmission=optsys.planes[0].amplitude,
    opd=optsys.planes[0].opd,
    pixelscale=optsys.planes[0].pixelscale,
    name=optsys.planes[0].name,
    oversample=optsys.planes[0].oversample
))

morphine_niriss.add_inversion(index=1, axis="y", name=optsys.planes[1].name)

# JWST internal WFE error
morphine_niriss.add_pupil(morphine.ArrayOpticalElement(
    transmission=optsys.planes[2].amplitude,
    opd=optsys.planes[2].opd,
    pixelscale=optsys.planes[2].pixelscale,
    name=optsys.planes[2].name,
    oversample=optsys.planes[2].oversample
))
# NRM
morphine_niriss.add_pupil(morphine.ArrayOpticalElement(
    transmission=optsys.planes[3].amplitude,
    opd=optsys.planes[3].opd,
    pixelscale=optsys.planes[3].pixelscale,
    name=optsys.planes[3].name,
    oversample=optsys.planes[3].oversample
))

morphine_niriss.add_detector(
    pixelscale=optsys.planes[-1].pixelscale.value, # does not support astropy units
    fov_arcsec=optsys.planes[-1].fov_arcsec.value,
    oversample=optsys.planes[-1].oversample,
    name=optsys.planes[-1].name
)

morphine_niriss.propagate(morphine_niriss.input_wavefront())
optsys.propagate().display()

