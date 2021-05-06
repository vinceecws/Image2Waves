import numpy as np
import cv2
import argparse
import datetime
import os.path as path
from matplotlib import pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='Directory for image to be transformed')
    parser.add_argument('threshold_min', nargs='?', default=100, type=int, help='Min threshold for hysteresis (default: %(default)s)')
    parser.add_argument('threshold_max', nargs='?', default=300, type=int, help='Max threshold for hysteresis (default: %(default)s)')
    parser.add_argument('--save', help='Folder to save transformed image')
    args = parser.parse_args()

    img = cv2.imread(args.dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(img, args.threshold_min, args.threshold_max)

    if args.save:
        cv2.imwrite(path.join(args.save, 'canny_img_{}.png'.format(datetime.datetime.now())), canny)

    plt.subplot(121),plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(canny, cmap='gray')
    plt.title('Canny Image'), plt.xticks([]), plt.yticks([])
    plt.show()
