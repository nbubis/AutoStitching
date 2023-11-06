import argparse
from functools import partial
from multiprocessing import Pool, cpu_count
import pathlib
import cv2
import numpy as np
import skimage.transform as transform


def mp_parallel_map(func, inputs, processes_count=-1):
    processes_count = processes_count if processes_count > 0 else cpu_count()
    processes_count = min(processes_count, len(inputs))
    with Pool(processes=processes_count) as pool:
        return list(pool.map(func, inputs))


def one_true(iterable):
    it = iter(iterable)
    return any(it) and not any(it)


def read_rescale_write(
    image_path: pathlib.Path,
    output_folder: pathlib.Path,
    num_pixels: int,
    rescale: float,
):
    scale = rescale if rescale else float(num_pixels) / max(img.shape[:2])
    img = cv2.imread(str(image_path))
    rescaled_img = transform.rescale(
        img,
        scale,
        channel_axis=2,
        anti_aliasing=True,
        preserve_range=True
    ).astype(np.uint8)
    cv2.imwrite(str(output_folder / image_path.name), rescaled_img)
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=pathlib.Path, required=True, help="folder with tif files.")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=None, help="folder to write to.")
    parser.add_argument("-p", "--num_pixels", type=int, help="number of pixels for the bigger axis.")
    parser.add_argument("-r", "--rescale", type=float, help="rescale factor to rescale the image.")
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    if not args.output: args.output: pathlib.Path = args.input
    args.output.mkdir(exist_ok=True, parents=True)
    
    assert one_true((args.num_pixels, args.rescale)), "you need to choose either number of "\
                                                      "pixels or rescale factor but not both"

    p_read_rescale_write = partial(read_rescale_write,
            output_folder=args.output,
            num_pixels=args.num_pixels,
            rescale=args.rescale)
    
    mp_parallel_map(p_read_rescale_write, args.input.glob('*'))
        
