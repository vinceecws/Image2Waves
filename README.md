# Image2Waves
Work in progress! This project is for participation in the Art of Neuroscience competition. Basically, the aim is to recreate portraits of people by simulating EEG waveforms.

To give it a go, clone the repo, then all you need to start playing around with converting images to waveforms is `python image2waves.py [...args]`!

For information on the possible arguments, do `python image2waves.py -h`, or refer to [**What are the parameters?**](#what-are-the-parameters)

### A short example to get you started:

Let's assume that I have cloned the repository and have already prepared an image in the same directory which is named `image.jpeg` (any other commonly used extensions that can be loaded by OpenCV works). So `image.jpeg` looks like this:

<p align="center">
  <img width="700" alt="original image" src="https://user-images.githubusercontent.com/19466657/119424906-621ba680-bcd4-11eb-92cd-35221824f21a.jpeg">
</p>

*Image taken from: https://www.aestheticsurgicalarts.com/aesthetic-surgical-arts/close-up-of-beautiful-woman-face/*

I pull up the shell, navigate to the current directory and do the following: 
```
python image2waves.py image2.jpeg --slice-height 50 --threshold 20 --factor 10 --filter sobel 3 3 --verbose --save
``` 
(The parameters will be explained shortly.)

Once the magic happens, you will observe 3 things: 
1) The shell window will print a summary report that looks like this:
<p align="center">
  <img width="700" alt="screenshot of summary report in shell" src="https://user-images.githubusercontent.com/19466657/119425889-339ecb00-bcd6-11eb-91a5-d2be31473726.png">
</p>

2) A plot will pop-up looking like this:
<p align="center">
  <img width="700" alt="plots of results" src="https://user-images.githubusercontent.com/19466657/119425988-61840f80-bcd6-11eb-81f7-c9f5fb8fa8ca.png">
</p>

The plot will show a side-by-side comparison of the before-and-after, plus the overlay.

3) And there will be a new image saved in the current directory named something like `image2waves_2021-05-24 21/18/59.838261.png`, which looks like this:
<p align="center">
  <img width="700" alt="waveform reconstructed image" src="https://user-images.githubusercontent.com/19466657/119426081-98f2bc00-bcd6-11eb-8244-22de7b3ed3b0.png">
</p>

*Note: In some cases, it's possible that you might see a blank image here. That's because the waveform image is generated as an RGBA with opacity 0 everywhere except for the pixels that form the waves (which have the color **white** and opacity at 255). Which means that the waves could be too thin to be observed without zooming-in, or that the default background color of your image viewer happens to be white. If those are not the cases here, then I suggest you to tweak the filtering parameters or the reconstruction parameters. Specifically, try increasing the density of the waveforms by reducing slice_height.*

## What are the parameters?
The algorithm is formed by a pipeline of generally two major blocks: **the feature extraction and the reconstruction**.
</br>
</br>
### Feature Extraction Parameters
For simplicity's sake, and because this part of the algorithm is not the spotlight here, I've only included two stock edge-extraction algorithms provided in the OpenCV library. Namely, the Sobel filter and the Canny Edge Detector.

The Sobel filter has two tweakable parameters: kernel_size_x and kernel_size_y, which are respectively the size (in pixels) of the x and y kernels. Detailed explanation of the OpenCV implementation here: https://docs.opencv.org/4.5.1/d2/d2c/tutorial_sobel_derivatives.html

The Canny Edge Detector similarly has two tweakable parameters: threshold_min and threshold_max, which represent respectively the minimum value and the maximum value of the thresholds used in the non-maxima supression stage of the edge detector. More info on the OpenCV implementation: https://docs.opencv.org/4.5.1/da/d22/tutorial_py_canny.html

The `--filter` command-line argument pertains to these filters and their parameters. In the above short example, I did `--filter sobel 3 3`, which means that the original image had its edges extracted with the **Sobel** filter and their kernel sizes were both **3** pixels. If you wish to go with the default kernel sizes, `--filter sobel` will do the trick. Omission of the `--filter` argument will bypass the feature extraction block and that the original image will be used for reconstruction (might yield subpar results). 

Parameters to the Canny Edge Detector is formatted in the same way as described in the case of the Sobel filter. `--filter canny 100 300` will produce a Canny edge image with non-maxima suppression minimum threshold of 100 and maximum threshold of 300.
</br>
</br>
### Wave Reconstruction Parameters
After feature extraction comes the reconstruction of the resultant image into waveforms. 
A brief explanation of the reconstruction algorithm goes like this:
1. The image is sliced along the y-axis into N equal parts (possibly with a remainder slice that is slightly smaller than the rest)
2. These vertical slices are downsampled along the x-axis (you can imagine it as going from left to right, 1 pixel is sampled for every K pixels)
3. For each x in the downsampled slice, the pixel column at x is weight-averaged to yield the mean height of the greatest pixel intensity.
4. With these points of interest, the wave reconstruction is performed by upsampling the points back to the original image width using Fourier analysis. Notes here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html
5. Each vertical slice will yield a waveform that best describes its general vicinity. They are then stacked back together to produce the final reconstructed image.

With the above as context, the `--slice-height` argument controls the height of the vertical slices (in pixels). 
The `--factor` argument is for the downsampling factor prior to weight-averaging the pixel columns, and subsequently the upsampling factor used in the wave reconstruction. A factor of 10 means 1 pixel is sampled for every 10 pixels.
Finally, a minute detail: the `--threshold` argument is used in the intensity thresholding after feature extraction and before reconstruction, in order to suppress any noise or artefact produced by the Sobel or Canny filters. The value for the parameter is scaled 0-255, whereby intensities lower than the threshold are set to 0.
</br>
</br>
### Additional Utility Parameters
To increase utility and comprehensability of the results, I have included the `--verbose` flag, which, when is set will print a brief summary report of the algorithm run.
Additionally, to save the resultant image, you can set the `--save` flag, which will save the image in the current working directory as `image2waves_[datetime].png`. The .png image is in RGBA, where the only non-zero, non-transparent pixels are the ones forming the waves.
