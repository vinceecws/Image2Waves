from matplotlib import pyplot as plt
import torch
import scipy.signal as signal
import cv2
import argparse
import time
import datetime
import os
from collections import namedtuple

import warnings
warnings.filterwarnings("ignore", category=UserWarning) #Temporary fix

filter_types = ['sobel', 'canny']

def filter_image(image, filter, *args):
    assert filter in filter_types, "Invalid filter type"

    if filter == 'sobel':
        kernel_size_x = args[0] if len(args) > 0 else 3
        kernel_size_y = args[1] if len(args) > 1 else 3

        if kernel_size_x not in [1, 3, 5, 7] or kernel_size_y not in [1, 3, 5, 7]:
            raise ValueError('Sobel kernel size must be either one of 1, 3, 5 or 7')

        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size_x, scale=1)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size_y, scale=1)
        abs_x = cv2.convertScaleAbs(sobel_x)
        abs_y = cv2.convertScaleAbs(sobel_y)
        sobel = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5,0) #Overlay x and y

        _ = namedtuple('Sobel', ['kernel_size_x', 'kernel_size_y'])
        params = _(kernel_size_x, kernel_size_y)

        return sobel, params

    elif filter == 'canny':
        threshold_min = args[0] if len(args) > 0 else 100
        threshold_max = args[1] if len(args) > 1 else 300
        canny = cv2.Canny(image, threshold_min, threshold_max)

        _ = namedtuple('Canny', ['threshold_min', 'threshold_max'])
        params = _(threshold_min, threshold_max)

        return canny, params

def load_image_and_preprocess(imdir, args_filter):

    im = cv2.imread(imdir)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    if args_filter is not None:
        im, filter_params = filter_image(im, args_filter[0], *[int(arg) for arg in args_filter[1:]])
        return torch.tensor(im), filter_params

    else:
        return torch.tensor(im)


def print_summary(start_time, slice_height, threshold, factor, filter_params=None):
    print('============================================================================================================')
    print('Image2Waves Summary:')
    print('Time elapsed: {:.2f}ms'.format((time.time() - start) * 1000))
    print()

    if filter_params is not None:
        print('Filter parameters:')
        print('     Type: {}'.format(type(filter_params).__name__))
        for param, value in filter_params._asdict().items():
            print('     {}: {}'.format(param, value))
        print()

    print('Wave reconstruction parameters:')
    print('     Slice Height: {}'.format(slice_height))
    print('     Threshold: {}'.format(threshold))
    print('     Factor: {}'.format(factor))
    print('============================================================================================================')

def convert_to_points(imslice, threshold=None, bucketize=False, factor=1):

    if threshold is not None and threshold >= 0 and threshold <= 255:
        imslice = torch.nn.functional.threshold(imslice, threshold, 0) #Filter out noise (Non-maxima suppression)

    ############################################################################################################
    newim = torch.nn.functional.interpolate(imslice.unsqueeze(0).unsqueeze(0), scale_factor=(1, 1 / factor)).squeeze() #Downsample with interpolation
    ############################################################################################################

    #Generate height-weighted grid for each column
    height = torch.arange(-(newim.shape[0] // 2), newim.shape[0] // 2 + newim.shape[0] % 2)
    height = torch.unsqueeze(height, 1).flip(0)
    height = torch.tile(height, (newim.shape[1],))

    #Weight pixels and select best point for each column
    points = (newim / newim.max())
    points = torch.sum(points * height, dim=0) / torch.sum(points, dim=0) #Height-weighted average among nonzero pixels
    points = points.nan_to_num(nan=0.0) #After division, completely empty columns will result in NaN due to sum = 0 / nonzero_pixel_count = 0, so convert NaN to 0.0

    ############################################################################################################
    points = torch.tensor(signal.resample(points, imslice.shape[1])).clamp(height.min(), height.max()) #Upsample using Fourier method and clamp any over/undershoot
    ############################################################################################################

    if bucketize:
        points = torch.Tensor.int(torch.round(points)) #Convert real-numbered points to nearest-integer pixel buckets

    return points

def image2waves(image, slice_height, threshold=None, factor=4):
    '''
    Image2Waves
    Steps:
        1) Extract significant features from image
        2) Downsample along horizontal axis of the feature image (width) by the given factor
        3) Convert each horizontal slice to set of (x, y) points where all x are unique
        4) Upsample generated points using Fourier method to create waveforms
        5) Plot waveforms along each horizontal slice

    '''

    if not isinstance(image, torch.Tensor):
        im = torch.tensor(image)
    else:
        im = image.clone()

    assert im.dim() == 2, "Image must only contain a single channel, only width and height dimensions are allowed. Image is assumed to be H x W."

    assert slice_height > 0 and slice_height <= im.shape[0], "Slice height must be any integer > 0 and <= image height"

    newim = torch.zeros_like(im)

    divisions = im.shape[0] // slice_height
    remainder_height = im.shape[0] % slice_height

    for start_ind in torch.arange(0, slice_height * divisions, slice_height):
        points = convert_to_points(im[start_ind : start_ind + slice_height, :], threshold=threshold, bucketize=True, factor=factor)
        points = (slice_height - 1) - (points + slice_height // 2) #Invert points and y-transform to center
        newim[start_ind : start_ind + slice_height, :] = torch.nn.functional.one_hot(points.long(), slice_height).transpose(1, 0) * 255

    if remainder_height > 0:
        points = convert_to_points(im[slice_height * divisions :, :], threshold=threshold, bucketize=True, factor=factor)
        points = (remainder_height - 1) - (points + remainder_height // 2) #Invert points and y-transform to center
        newim[slice_height * divisions :, :] = torch.nn.functional.one_hot(points.long(), remainder_height).transpose(1, 0) * 255

    return newim


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('imdir', help='Directory for image to be transformed')
    parser.add_argument('--slice_height', nargs=1, default=50, type=int, help='Height (in pixels) of slices of the original image to be reconstructed as waves. Must be a positive integer <= image height  (default: %(default)s)')
    parser.add_argument('--threshold', nargs=1, default=20, type=int, help='Noise filtering. If given, the intensity threshold below which pixels will be zeroed, such that 0 <= threshold <= 255. Values beyond range will be ignored.')
    parser.add_argument('--factor', nargs=1, default=10, type=int, help='The factor that will be used to downsample the horizontal axis (width) of the image, and upsampled using the Fourier method. (default: %(default)s)')
    parser.add_argument('--filter', nargs='+', default=None, help='Filter that is used prior to wave reconstruction. First argument must be one of {}. Subsequent arguments are: \
        For Sobel, arg1 and arg2 are the size (in pixels) of the horizontal and vertical kernels, respectively.\
        For Canny, arg1 and arg2 are the minimum and maximum threshold values, respectively, used in non-maxima suppression'.format(filter_types))
    parser.add_argument('--verbose', action='store_true', help='Flag. Prints summary report upon completion.')
    parser.add_argument('--save', action='store_true', help='Flag. Save reconstructed image in current directory. Filename will be in the format of image2waves_dateandtime.png')

    args = parser.parse_args()

    im, filter_params = load_image_and_preprocess(args.imdir, args.filter)

    start = time.time()
    newim = image2waves(im, args.slice_height, threshold=args.threshold, factor=args.factor)

    if args.verbose:
        print_summary(start, args.slice_height, args.threshold, args.factor, filter_params=filter_params if filter_params else None)

    if args.save:
        cv2.imwrite(os.path.join(os.getcwd(), 'image2waves_{}.png'.format(datetime.datetime.now())), newim.numpy())

    plt.imshow(newim, cmap='gray')
    plt.show()
