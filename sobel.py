import numpy as np
import cv2
import argparse
import datetime
import os
from matplotlib import pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='Directory for image to be transformed')
    parser.add_argument('kernel_size_x', nargs='?', default=3, type=int, help='Sobel kernel size for X, values from [1, 3, 5, 7] (default: %(default)s)')
    parser.add_argument('kernel_size_y', nargs='?', default=3, type=int, help='Sobel kernel size for Y, values from [1, 3, 5, 7] (default: %(default)s)')
    parser.add_argument('--save', action='store_true', help='Flag. Saves transformed image in current directory. Filename will be in the format of sobel_img_dateandtime.png')
    args = parser.parse_args()

    if args.kernel_size_x not in [1, 3, 5, 7] or args.kernel_size_y not in [1, 3, 5, 7]:
        raise ValueError('Sobel kernel size must be either one of 1, 3, 5 or 7')

    im = cv2.imread(imdir, 0) #Read as grayscale
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=args.kernel_size_x, scale=1)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=args.kernel_size_y, scale=1)
    abs_x = cv2.convertScaleAbs(sobel_x)
    abs_y = cv2.convertScaleAbs(sobel_y)
    sobel = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5,0) #Overlay x and y

    if args.save:
        cv2.imwrite(os.path.join(os.getcwd(), 'sobel_img_{}.png'.format(datetime.datetime.now())), sobel)

    plt.subplot(121),plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(sobel, cmap='gray')
    plt.title('Sobel Image'), plt.xticks([]), plt.yticks([])
    plt.show()