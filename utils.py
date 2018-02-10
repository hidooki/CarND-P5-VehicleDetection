"""This file contains utilities for image processing"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import os


def read_image(image_name):
    """Extension agnostic image reader. Returns RGB image"""
    bgr = cv2.imread(image_name)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def plot_pairs(images, titles=None, figsize=None, cmap='gray', labelsize=5):
    """Plot a batch of images, two on a line"""
    lines = (len(images) + 1) // 2

    plt.interactive(True)
    fig, axes = plt.subplots(lines, 2, figsize=figsize)

    if lines == 1:
        axes = [axes]

    if titles is None:
        titles = len(images) * ['']

    for l in range(lines):
        for c in range(2):
            axes[l][c].imshow(images[2 * l + c], cmap=cmap)
            axes[l][c].set_title(titles[2 * l + c])
            axes[l][c].locator_params(nbins=10, axis='x')
            axes[l][c].tick_params(labelsize=labelsize)

    fig.tight_layout()

    return fig


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy


def slide_window(
        image,
        x_start_stop=[None, None],
        y_start_stop=[None, None],
        xy_window=(64, 64),
        xy_overlap=(0.5, 0.5)):
    """Slide window across given image

    Input
        image: original image
        x_start_stop: search area box x boundaries
        y_start_stop: search area box y boundaries
        xy_window: size of search window
        xy_overlap: percentage overlap between adjacent windows

    Ouput
        List of windows' coordinates [(x_start, y_start), (x_end, y_end]"""

    ny, nx, _ = image.shape

    # If x and/or y start/stop positions not defined, set to image size
    for start_stop, n in zip([x_start_stop, y_start_stop], [nx, ny]):
        start, stop = start_stop
        start_stop[0] = 0 if start is None else start
        start_stop[1] = n if stop is None else stop

    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)

    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            window_list.append(((startx, starty), (endx, endy)))

    return window_list


def extract_frames(clip, folder):
    """Given video clip, extract frames in folder"""
    frames = clip.iter_frames()
    for i, frame in enumerate(frames):
        fname = os.path.join(folder, 'frame_%s.jpg' % i)
        # cv2 much faster than matplotlib.image, but adjust for BGR default..
        cv2.imwrite(fname, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':

    image = read_image('test_images/test1.jpg')
    # image = read_image('sandbox/bbox-example-image.jpg')
    focus = draw_boxes(cv2.resize(image, (1280, 720)), [[(0, 400), (1280, 656)]])
    plt.imshow(focus)
    plt.savefig('output_images/search_window.jpg')
