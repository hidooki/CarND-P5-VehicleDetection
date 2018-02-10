"""Car detection pipeline"""
from collections import deque
import cv2
import glob
from itertools import product
from matplotlib import pyplot as plt
from moviepy.editor import VideoFileClip
import numpy as np
import pickle
from scipy.ndimage.measurements import label
import os

from utils import draw_boxes, read_image, plot_pairs
from features import color_convertor, bin_spatial, get_hog_features, color_hist


# Load model params
params = pickle.load(open('params/params.p', 'rb'))
svc = pickle.load(open('params/classifier.p', 'rb'))
X_scaler = pickle.load(open('params/feature_scaler.p', 'rb'))
print('Loaded parameters: ', params)


# Define a single function that can extract features
# using hog sub-sampling and make predictions

def find_cars(
        image,
        y_start_stop,
        scale,
        clf,
        X_scaler,
        color_space,
        spatial_size,
        hist_bins,
        orient,
        pix_per_cell,
        cell_per_block,
        channels):

    # draw_img = np.copy(image)
    img = image.astype(np.float32) / 255

    ystart, ystop = y_start_stop
    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = color_convertor(color_space)(img_tosearch)

    imshape = ctrans_tosearch.shape
    nx, ny = np.int(imshape[1] / scale), np.int(imshape[0] / scale)
    ctrans_tosearch = cv2.resize(ctrans_tosearch, (nx, ny))

    # Define blocks and steps as above
    nxblocks = (ctrans_tosearch.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ctrans_tosearch.shape[0] // pix_per_cell) - cell_per_block + 1
    # nfeat_per_block = orient * cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    x_stub = np.mod(ctrans_tosearch.shape[1], pix_per_cell * cell_per_block)

    # Compute HOG features for the entire image, subsample later
    hog_features_search = []
    for c in channels:
        hog = get_hog_features(
            ctrans_tosearch[:, x_stub:, c],
            orient,
            pix_per_cell,
            cell_per_block,
            vis=False,
            feature_vec=False)
        hog_features_search.append(hog)

    box_list = []
    for xb, yb in product(range(nxsteps), range(nysteps)):
        ypos = yb * cells_per_step
        xpos = xb * cells_per_step

        # Extract HOG for this patch
        hog_feats = []
        for hog in hog_features_search:
            feats = hog[ypos:ypos + nblocks_per_window,
                        xpos:xpos + nblocks_per_window]
            hog_feats.append(feats.ravel())
        hog_features = np.hstack(hog_feats)

        xleft = x_stub + xpos * pix_per_cell
        ytop = ypos * pix_per_cell

        # Extract the image patch
        subimg = cv2.resize(
            ctrans_tosearch[ytop:ytop + window,
                            xleft:xleft + window],
            (64, 64))

        # Get color features
        spatial_features = bin_spatial(subimg, size=spatial_size)
        hist_features = color_hist(subimg, nbins=hist_bins)

        # Scale features and make a prediction
        features = np.hstack((spatial_features, hist_features, hog_features))
        test_features = X_scaler.transform(features.reshape(1, -1))

        # test_features = X_scaler.transform(
        #     np.hstack((spatial_features, hist_features)).reshape(1, -1))

        test_prediction = clf.predict(test_features)

        if test_prediction == 1:
            xbox_left = np.int(xleft * scale)
            ytop_draw = np.int(ytop * scale)
            win_draw = np.int(window * scale)

            box = ((xbox_left, ytop_draw + ystart),
                   (xbox_left + win_draw, ytop_draw + win_draw + ystart))

            # cv2.rectangle(draw_img, box[0], box[1], (0, 0, 255), 6)
            box_list.append(box)

    return box_list


def find_cars_multiple_scales(
        image,
        clf,
        X_scaler,
        color_space,
        spatial_size,
        hist_bins,
        orient,
        pix_per_cell,
        cell_per_block,
        channels):

    y_zones = [
        # ((400, 464), .75),

        ((400, 400 + 2 * 64), 1),
        ((408, 408 + 2 * 64), 1),
        ((416, 416 + 2 * 64), 1),

        ((400, 496), 1.5),
        ((408, 408 + 96), 1.5),
        ((416, 416 + 96), 1.5),

        # ((400, 400 + 2 * 64), 2),
        ((432, 432 + 2 * 64), 2),
        ((656 - 3 * 64, 656 - 64), 2),

        ((656 - 3 * 64, 656), 3)

    ]

    boxes = []
    for y_start_stop, scale in y_zones:
        boxes += find_cars(
            image,
            y_start_stop,
            scale,
            clf,
            X_scaler,
            color_space,
            spatial_size,
            hist_bins,
            orient,
            pix_per_cell,
            cell_per_block,
            channels)

    return boxes


# ---- Heatmap utils from lesson ----

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(image, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(image, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return image


def make_pipeline(params, n_integrate=10, threshold=3):

    heat_all = deque(maxlen=n_integrate)

    def pipeline(image):

        box_list = find_cars_multiple_scales(
            image,
            svc,
            X_scaler,
            params['color_space'],
            params['spatial_size'],
            params['histbin'],
            params['orient'],
            params['pix_per_cell'],
            params['cell_per_block'],
            params['channels'])

        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        add_heat(heat, box_list)

        heat_all.append(heat)
        total_heat = np.sum(np.stack(heat_all, axis=2), axis=2)

        # Apply threshold to help remove false positives
        heat = apply_threshold(total_heat, threshold)
        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(image), labels)

        return draw_img

    return pipeline


# Plotting functions produce images for the writeup

def plot_boxes(imgs):

    draw_imgs = []
    for img in imgs:
        image = read_image(img)
        box_list = find_cars_multiple_scales(
            image,
            svc,
            X_scaler,
            params['color_space'],
            params['spatial_size'],
            params['histbin'],
            params['orient'],
            params['pix_per_cell'],
            params['cell_per_block'],
            params['channels'])

        draw_imgs.append(draw_boxes(np.copy(image), box_list))

    plot_pairs(draw_imgs, figsize=(8, 10), cmap=None, labelsize=8)
    plt.savefig('output_images/boxes.jpg')


def plot_boxes_and_heat(imgs):

    box_heat, titles = [], []
    for img in imgs:
        image = read_image(img)
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)

        box_list = find_cars_multiple_scales(
            image,
            svc,
            X_scaler,
            params['color_space'],
            params['spatial_size'],
            params['histbin'],
            params['orient'],
            params['pix_per_cell'],
            params['cell_per_block'],
            params['channels'])

        draw_img = draw_boxes(np.copy(image), box_list)

        # draw_img = draw_boxes(image, box_list)
        add_heat(heat, box_list)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 1)
        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        box_heat += [draw_img, heatmap]

        title = os.path.basename(img)
        titles += [title, title + ' heatmap']

    plot_pairs(box_heat, figsize=(8, 10), cmap='hot', labelsize=8)
    plt.savefig('output_images/boxes_and_heat.jpg')


if __name__ == '__main__':

    # Boxes and heat plot

    # imgs = glob.glob('frames/frame*.jpg')
    # imgs = glob.glob('test_images/test*.jpg')
    # plot_boxes(imgs)

    # plot_boxes_and_heat(imgs)

    # Run pipeline on project video

    # pipe = make_pipeline(params, n_integrate=1, threshold=1)
    #
    # image = read_image('test_images/test6.jpg')
    # draw_img = pipe(image)
    # plt.imshow(draw_img)
    # plt.savefig('output_images/cars_found.jpg')

    # Video processing

    pipe = make_pipeline(params, n_integrate=10, threshold=5)
    clip_name = "project_video.mp4"

    clip = VideoFileClip(clip_name)

    out_clip = clip.fl_image(pipe)
    out_clip.write_videofile(clip_name.replace('.', '_result.'), audio=False)
