"""Feature extraction functionality"""
import glob
import cv2
import numpy as np

from matplotlib import pyplot as plt
from skimage.feature import hog

from utils import read_image, plot_pairs


def color_convertor(color_space='RGB'):
    """Returns convertor from RGB to desired color space

    color_space: RGB, HSV, HLS, LUV, YUV, YCrCb"""
    convertor = {'HSV': cv2.COLOR_RGB2HSV,
                 'HLS': cv2.COLOR_RGB2HLS,
                 'LUV': cv2.COLOR_RGB2LUV,
                 'YUV': cv2.COLOR_RGB2YUV,
                 'YCrCb': cv2.COLOR_RGB2YCrCb}

    if color_space.upper() == 'RGB':
        def conv_func(img):
            return img
    else:
        def conv_func(img):
            return cv2.cvtColor(img, convertor[color_space])

    return conv_func


# def bin_spatial(image, size=(32, 32)):
#     """Compute a features vector obtained by spatial binning the image"""
#     return cv2.resize(image, size).ravel()

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(image, nbins=32, bins_range=None):
    """Computes a features vector by concating the histograms of all
    color channels"""

    histograms = []
    for ch in range(3):
        hist = np.histogram(image[:, :, ch], bins=nbins, range=bins_range)
        histograms.append(hist[0])

    return np.concatenate(histograms)


def get_hog_features(
        image,
        orient,
        pix_per_cell,
        cell_per_block,
        vis=False,
        feature_vec=True):
    """Extract Histogram of Oriented Gradient features

    Input
        orient: gradient orientation
        pix_per_cell: specifies cell dimension
        cell_per_block: bock dimension
        vis: If True, the HOG image will be returned as a second output
        feature_vec: If True, feature vector returned
    Output
        HOG features vector, (HOG image if vis is True)
    """

    if vis:
        features, hog_image = hog(
            image,
            orientations=orient,
            pixels_per_cell=(pix_per_cell, pix_per_cell),
            cells_per_block=(cell_per_block, cell_per_block),
            transform_sqrt=True,
            visualise=vis,
            feature_vector=feature_vec,
            block_norm='L1')
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(
            image,
            orientations=orient,
            pixels_per_cell=(pix_per_cell, pix_per_cell),
            cells_per_block=(cell_per_block, cell_per_block),
            transform_sqrt=True,
            visualise=vis,
            feature_vector=feature_vec,
            block_norm='L1')
        return features


def feature_extractor(
        image,
        resize=(64, 64),
        color_space='RGB',
        spatial_size=(32, 32),
        hist_bins=32,
        hist_range=None,
        hog_orient=9,
        hog_pix_per_cell=8,
        hog_cell_per_block=2,
        hog_channels=range(3),
        spatial_features=False,
        color_features=False,
        hog_features=True):
    """Convert image to(smaller) size in the specified color space

    Input
        resize: proceed after resizing the image
        color_space: RGB, HSV, HLS, LUV, YUV or YCrCb

    Output
        Feature vector obtained by ravelling the smaller, resized image
    """

    if resize is not None:
        image_res = cv2.resize(image, resize)
    else:
        image_res = image

    color_converted = color_convertor(color_space)(image_res)

    features = []
    if spatial_features:
        features.append(bin_spatial(color_converted, size=spatial_size))

    if color_features:
        features.append(color_hist(
            color_converted,
            nbins=hist_bins,
            bins_range=hist_range))

    if hog_features:
        for ch in hog_channels:
            features.append(get_hog_features(
                color_converted[:, :, ch],
                orient=hog_orient,
                pix_per_cell=hog_pix_per_cell,
                cell_per_block=hog_cell_per_block,
                vis=False,
                feature_vec=True))

    return np.concatenate(features)


if __name__ == '__main__':

    cars = glob.glob('images/vehicles/KITTI*/*.png')
    notcars = glob.glob('images/non-vehicles/Extra*/*.png')

    acar = read_image(cars[100])
    anotcar = read_image(notcars[100])

    plot_pairs([acar, anotcar], ['A car', 'Not a car'], labelsize=7)
    plt.savefig('output_images/car_not_car.jpg')

    # HOG plot

    color_space = 'YCrCb'
    for what, img in zip(['A car', 'Not a car'], [acar, anotcar]):
        image = color_convertor(color_space)(img)

        pics, titles = [], []
        for ch in range(3):
            hog_feats, hog_img = get_hog_features(
                image[:, :, ch],
                orient=11,
                pix_per_cell=8,
                cell_per_block=2,
                vis=True,
                feature_vec=True)

            pics += [image[:, :, ch], hog_img]
            channel = color_space[ch]
            titles += ['%s CH-%s' % (what, ch), '%s CH-%s HOG' % (what, ch)]

        plot_pairs(pics, titles, figsize=(6, 8), cmap='gray', labelsize=7)
        plt.savefig('output_images/%s_hog_visualisation.jpg' % what.replace(' ', '_'))
