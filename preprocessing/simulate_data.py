from ExoRIM.simulated_data import CenteredImagesv1
from PIL import Image
from ExoRIM.utilities import convert_to_8_bit
import tarfile
import pickle
import glob, os


def create_and_save_data(datadir, meta_data):
    images = meta_data.generate_epoch_images()
    for i, image in enumerate(images):  # iterate over first dimension
        image = convert_to_8_bit(image)
        im = Image.fromarray(image[:, :, 0], mode="L") # for grau images, channels = 1
        im.save(os.path.join(datadir, f"image{i:03}.png"))
    with open(os.path.join(datadir, "meta_data.pickle"), "wb") as f:
        pickle.dump(meta_data, f)
    with tarfile.open(os.path.join(datadir, "images.tar.gz"), "x:gz") as tar:
        for file in glob.glob(os.path.join(datadir, "*.png")):
            tar.add(file)
    return images


if __name__ == "__main__":
    from argparse import ArgumentParser
    from datetime import datetime
    parser = ArgumentParser()
    parser.add_argument("-d", "--dir_name", type=str, default=datetime.now().strftime("%y-%m-%d_%H-%M-%S"), required=False)
    parser.add_argument("-n", "--number_images", type=int, default=100)
    parser.add_argument("-p", "--pixels", type=int, default=32)
    parser.add_argument("-c", "--contrast", type=float, default=0.95)
    parser.add_argument("-s", "--sources", type=int, default=5)
    args = parser.parse_args()
    dirname = args.dir_name
    datadir = os.path.join(os.getcwd(), "data", dirname)
    meta_data = CenteredImagesv1(
        total_items=args.number_images,
        channels=1,
        pixels=args.pixels,
        highest_contrast=args.contrast,
        max_point_sources=args.sources
    )
    if not os.path.isdir(datadir):
        os.mkdir(datadir)
    create_and_save_data(datadir, meta_data)

