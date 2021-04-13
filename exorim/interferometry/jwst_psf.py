# import morphine
import webbpsf
import matplotlib.pyplot as plt

FILTERS_WHEEL = ["F277W", "F444W", "F356W", "F430M", "F380M", "F480M"]
# FILTERS_PUPIL = ["F200W", "F150W", "F140M", "F158M", "F115W", "F090W"] # not accessible since we use NRM
FILTERS = FILTERS_WHEEL 


def main(args):
    niriss = webbpsf.NIRISS()
    niriss.pupil_mask = "MASK_NRM"
    if args.filter in FILTERS:
        niriss.filter = args.filter
        niriss.calc_psf(oversample=args.oversample, outfile=f"../../data/psf/jwst_{args.filter}_{args.oversample}_psf.fits", display=args.display, overwrite=True)
        if args.display:
            plt.show()
    elif args.filter == "all":
        for f in FILTERS:
            niriss.filers = f
            niriss.calc_psf(oversample=args.oversample, outfile=f"models/psf/jwst_{f}_{args.oversample}_psf.fits", display=False, overwrite=True)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--filter", default="all", help="Filters to use when calculating psf, default is to calculate all psf w")
    parser.add_argument("--oversample", default=2, type=int, help="FFT oversampling factor, control the pixelsize of the simulated detector")
    parser.add_argument("--display", action="store_true", help="If a single filter is selected, display the psf")
    args = parser.parse_args()
    main(args)


# optsys = niriss.get_optical_system()
# morphine_niriss = morphine.OpticalSystem(name="ami_niriss")

# # entrance pupil
# morphine_niriss.add_pupil(morphine.ArrayOpticalElement(
    # transmission=optsys.planes[0].amplitude,
    # opd=optsys.planes[0].opd,
    # pixelscale=optsys.planes[0].pixelscale,
    # name=optsys.planes[0].name,
    # oversample=optsys.planes[0].oversample
# ))

# morphine_niriss.add_inversion(index=1, axis="y", name=optsys.planes[1].name)

# # JWST internal WFE error
# morphine_niriss.add_pupil(morphine.ArrayOpticalElement(
    # transmission=optsys.planes[2].amplitude,
    # opd=optsys.planes[2].opd,
    # pixelscale=optsys.planes[2].pixelscale,
    # name=optsys.planes[2].name,
    # oversample=optsys.planes[2].oversample
# ))
# # NRM
# morphine_niriss.add_pupil(morphine.ArrayOpticalElement(
    # transmission=optsys.planes[3].amplitude,
    # opd=optsys.planes[3].opd,
    # pixelscale=optsys.planes[3].pixelscale,
    # name=optsys.planes[3].name,
    # oversample=optsys.planes[3].oversample
# ))

# morphine_niriss.add_detector(
    # pixelscale=optsys.planes[-1].pixelscale.value, # does not support astropy units
    # fov_arcsec=optsys.planes[-1].fov_arcsec.value,
    # oversample=optsys.planes[-1].oversample,
    # name=optsys.planes[-1].name
# )

# morphine_niriss.propagate(morphine_niriss.input_wavefront())
# optsys.propagate().display()

