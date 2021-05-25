import numpy as np
import cv2
import argparse
import datetime
import os
from matplotlib import pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='Directory for image to be transformed')
    parser.add_argument('threshold_min', nargs='?', default=100, type=int, help='Min threshold for hysteresis (default: %(default)s)')
    parser.add_argument('threshold_max', nargs='?', default=300, type=int, help='Max threshold for hysteresis (default: %(default)s)')
    parser.add_argument('--save', action='store_true', help='Flag. Saves transformed image in current directory. Filename will be in the format of canny_img_dateandtime.png')
    args = parser.parse_args()

    im = cv2.imread(imdir, 0) #Read as grayscale
    canny = cv2.Canny(img, args.threshold_min, args.threshold_max)

    if args.save:
        cv2.imwrite(os.path.join(os.getcwd(), 'canny_img_{}.png'.format(datetime.datetime.now())), canny)

    plt.subplot(121),plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(canny, cmap='gray')
    plt.title('Canny Image'), plt.xticks([]), plt.yticks([])
    plt.show()
