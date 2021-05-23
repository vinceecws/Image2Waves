import torch
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import math
import cv2

def x_n(n, coeff, phi):

    expo = torch.zeros(coeff.shape[0], 2)
    expo[:, 1] = ((2 * np.pi) / coeff.shape[0]) * n * torch.arange(coeff.shape[0]) + phi

    return torch.mean(coeff * torch.exp(torch.view_as_complex(expo)))

def deconstruct_X(data):

    fourier = torch.fft.rfft(data)

    return fourier 

def reconstruct_X(data, threshold, order=None):

    # assert order is None or (order > 0 and order <= data.shape[0])
    assert threshold >= 0, "Threshold for high-coefficient-pass filter must be >= 0"

    # coeff = fourier.abs() # Magnitude of Fourier coefficients
    # phi = fourier.angle() # Angle of Fourier coefficients

    # if order is not None: #Extract the best descriptors
    #     best_ind = coeff[:coeff.shape[0] // 2].topk(order).indices
    #     print(best_ind)
    #     print(coeff)
    #     coeff = coeff[best_ind]
    #     phi = phi[best_ind]

    # return torch.tensor([x_n(i, coeff, phi).real for i in range(data.shape[0])]), fourier

    reconstructed = data.clone()
    reconstructed[torch.where(reconstructed.abs() < threshold)] = 0 #Zero out frequency components below a threshold
    reconstructed = torch.fft.irfft(reconstructed, data.numel()) #Inverse FFT for reconstruction

    return reconstructed

def plot_normal(mean=0.0, std=1.0, lower_bound=-3.0, upper_bound=3.0, num_points=100, to_torch=False):

    assert lower_bound < upper_bound, "Lower bound must be strictly < upper bound"

    x = np.linspace(lower_bound, upper_bound, num_points)
    y = stats.norm.pdf(x, mean, std)

    if to_torch:
        x = torch.tensor(x)
        y = torch.tensor(y)

    return x, y

def ed_noise_uniform(data, lower_bound=0.0, upper_bound=1.0):
    assert lower_bound < upper_bound, "Lower bound must be strictly < upper bound"
    
    return data + (upper_bound - lower_bound) * torch.rand(data.shape) + lower_bound

def convert_to_points(imslice, threshold=None, bucketize=False):

    if threshold is not None:
        imslice = torch.nn.functional.threshold(imslice, threshold, 0) #Filter out noise (Non-maxima suppression)

    #Generate height-weighted grid for each column
    height = torch.arange(-(imslice.shape[0] // 2), imslice.shape[0] // 2 + imslice.shape[0] % 2)
    height = torch.unsqueeze(height, 1).flip(0)
    height = torch.tile(height, (imslice.shape[1],))

    #Weight pixels and select best point for each column
    points = (imslice / imslice.max())
    points = torch.sum(points * height, dim=0) / torch.sum(points, dim=0) #Height-weighted average among nonzero pixels
    points = points.nan_to_num(nan=0.0) #After division, completely empty columns will result in NaN due to sum = 0 / nonzero_pixel_count = 0, so convert NaN to 0.0

    if bucketize:
        points = torch.Tensor.int(torch.round(points)) #Convert real-numbered points to nearest-integer pixel buckets

    return points

def image2waves(image, slice_height, threshold=None):
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

    for start_ind in np.arange(0, slice_height * divisions, slice_height):
        points = convert_to_points(im[start_ind : start_ind + slice_height, :], threshold=threshold, bucketize=True)
        points = (slice_height - 1) - (points + slice_height // 2) #Invert points and y-transform to center
        newim[start_ind : start_ind + slice_height, :] = torch.nn.functional.one_hot(points.long(), slice_height).transpose(1, 0) * 255

    if remainder_height > 0:
        points = convert_to_points(im[slice_height * divisions :, :], threshold=threshold, bucketize=True)
        points = (remainder_height - 1) - (points + remainder_height // 2) #Invert points and y-transform to center
        newim[slice_height * divisions :, :] = torch.nn.functional.one_hot(points.long(), remainder_height).transpose(1, 0) * 255

    return newim


if __name__ == '__main__':

    im = cv2.imread('image2_sobel.png')
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im = torch.tensor(im)

    newim = image2waves(im, 150, threshold=20)

    plt.imshow(newim, cmap='gray')
    plt.show()
