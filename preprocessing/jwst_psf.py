# import morphine
import webbpsf
import matplotlib.pyplot as plt
from tqdm import tqdm

FILTERS_WHEEL = ["F277W", "F444W", "F356W", "F430M", "F380M", "F480M"]
# FILTERS_PUPIL = ["F200W", "F150W", "F140M", "F158M", "F115W", "F090W"] # not accessible since we use NRM
FILTERS = FILTERS_WHEEL 

# ck04 table
spectral_types = [
            "O3V",
            "O5V",
            "O6V",
            "O8V",
            "O5I",
            "O6I",
            "O8I",
            "B0V",
            "B3V",
            "B5V",
            "B8V",
            "B0III",
            "B5III",
            "B0I",
            "B5I",
            "A0V",
            "A5V",
            "A0I",
            "A5I",
            "F0V",
            "F5V",
            "F0I",
            "F5I",
            "G0V",
            "G5V",
            "G0III",
            "G5III",
            "G0I",
            "G5I",
            "K0V",
            "K5V",
            "K0III",
            "K5III",
            "K0I",
            "K5I",
            "M0V",
            "M2V",
            "M5V",
            "M0III",
            "M0I",
            "M2I"]

# phoenix table, used int JWST ETCs
spectral_types2 = [
            "O3V",
            "O5V",
            "O7V",
            "O9V",
            "B0V",
            "B1V",
            "B3V",
            "B5V",
            "B8V",
            "A0V",
            "A1V",
            "A3V",
            "A5V",
            "F0V",
            "F2V",
            "F5V",
            "F8V",
            "G0V",
            "G2V",
            "G5V",
            "G8V",
            "K0V",
            "K2V",
            "K5V",
            "K7V",
            "M0V",
            "M2V",
            "M5V",
            "B0III",
            "B5III",
            "G0III",
            "G5III",
            "K0III",
            "K5III",
            "M0III",
            "O6I",
            "O8I",
            "B0I",
            "B5I",
            "A0I",
            "A5I",
            "F0I",
            "F5I",
            "G0I",
            "G5I",
            "K0I",
            "K5I",
            "M0I",
            "M2I"]

def main(args):
    niriss = webbpsf.NIRISS()
    niriss.pupil_mask = "MASK_NRM"
    if args.filter in FILTERS:
        niriss.filter = args.filter
        for i in tqdm(range(10)):
            niriss.pupilopd = ('OPD_RevW_ote_for_NIRISS_requirements.fits.gz', i)
            for spectral in tqdm(spectral_types):
                src = webbpsf.specFromSpectralType(spectral, catalog="ck04")
                niriss.calc_psf(oversample=args.oversample, outfile=f"../data/psf/jwst_{args.filter}_{args.oversample}_psf_OPD{i+1:02}_{spectral}.fits", 
                    display=args.display, overwrite=True, normalize='last', source=src)
        if args.display:
            plt.show()
    elif args.filter == "all":
        for f in FILTERS:
            niriss.filers = f
            niriss.calc_psf(oversample=args.oversample, outfile=f"../data/psf/jwst_{f}_{args.oversample}_psf.fits", display=False, overwrite=True)


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

