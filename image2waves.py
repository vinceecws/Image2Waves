from matplotlib import pyplot as plt
import torch
import scipy.signal as signal
import cv2
import argparse
import time

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
        1) Find edges (and contours/depth)
        2) Convert each horizontal slice to set of (x, y) points where all x are unique
        3) Smooth points with Lagrange polynomial interpolation
        4) Plot curves (no discontinuity)

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
    parser.add_argument('slice_height', nargs='?', default=100, type=int, help='Height (in pixels) of slices of the original image to be reconstructed as waves. Must be a positive integer <= image height  (default: %(default)s)')
    parser.add_argument('threshold', nargs='?', default=-1, type=int, help='If given, the intensity threshold below which pixels will be zeroed, such that 0 <= threshold <= 255. Values beyond range will be ignored.')
    parser.add_argument('factor', nargs='?', default=1, type=int, help='The factor that will be used to downsample the horizontal axis (width) of the image, and upsampled using the Fourier method. (default: %(default)s)')

    args = parser.parse_args()

    im = cv2.imread(args.imdir)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im = torch.tensor(im)

    start = time.time()
    newim = image2waves(im, args.slice_height, threshold=args.threshold, factor=args.factor)
    print('Time elapsed: {:.2f}ms'.format((time.time() - start) * 1000))

    plt.imshow(newim, cmap='gray')
    plt.show()
