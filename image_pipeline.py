import glob
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
from skimage.feature import hog
from scipy.ndimage.measurements import label
from collections import deque
from moviepy.editor import VideoFileClip


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vector=True):
    # Define a function to return HOG features and visualization
    # If feature_vector is True, a 1D (flattened) array is returned.
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  block_norm= 'L2-Hys',
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualize=vis, feature_vector=feature_vector)
        return features, hog_image
    else:
        features = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  block_norm= 'L2-Hys',
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualize=vis, feature_vector=feature_vector)
        return features

def convert_color(img, conv='YCrCb'):
    # Define a function to convert color space
    if conv == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    if conv == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    if conv == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    if conv == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if conv == 'RGB':
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def bin_spatial(img, size=(16,16)):
    # Define a function to compute binned color features
    return cv2.resize(img, size).ravel()

def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Define a function to compute color histogram features
    channel1 = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2 = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3 = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((channel1[0], channel2[0], channel3[0]))
    return hist_features

def extract_features(imgs, color_space='BGR', spatial_size=(16, 16),
                     hist_bins=128, orient=9, pix_per_cell=8,
                     cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    features = []
    for img in imgs:
        img_features = []
        image = cv2.imread(img)
        if color_space != 'BGR':
            feature_image = convert_color(image, color_space)
        else:
            feature_image = np.copy(image)

        if spatial_feat:
            spatial_feature = bin_spatial(feature_image, spatial_size)
            img_features.append(spatial_feature)
        if hist_feat:
            hist_feature = color_hist(feature_image, hist_bins)
            img_features.append(hist_feature)
        if hog_feat:
            if hog_channel == 'ALL':
                hog_ch1 = get_hog_features(feature_image[:,:,0], orient,
                              pix_per_cell, cell_per_block, vis=False, feature_vector=True)
                hog_ch2 = get_hog_features(feature_image[:,:,1], orient,
                              pix_per_cell, cell_per_block, vis=False, feature_vector=True)
                hog_ch3 = get_hog_features(feature_image[:,:,2], orient,
                              pix_per_cell, cell_per_block, vis=False, feature_vector=True)
                hog_feature = np.concatenate((hog_ch1, hog_ch2, hog_ch3))
            else:
                hog_feature = get_hog_features(feature_image[:,:,hog_channel], orient,
                              pix_per_cell, cell_per_block, vis=False, feature_vector=True)
            img_features.append(hog_feature)
        features.append(np.concatenate(img_features))
    return features

def find_cars(img, ystart, ystop, color_space, scale, svc, X_scaler, orient,
              pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)
    # img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv=color_space)
    if scale != 1:
        image_shape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(image_shape[1]/scale), np.int(image_shape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Compute individual channel HOG features for the entire image
    # hog dimension = (nyblocks x nxblocks x cell_per_block x cell_per_block x orient)
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis=False, feature_vector=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis=False, feature_vector=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis=False, feature_vector=False)

    x_nblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    y_nblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1

    window_size = 64   # pixels
    block_per_window = (window_size // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2
    nx_step = 1 + (x_nblocks - block_per_window) // cells_per_step
    ny_step = 1 + (y_nblocks - block_per_window) // cells_per_step

    car_windows = []

    for yb in range(ny_step):
        for xb in range(nx_step):
            xpos = xb * cells_per_step
            ypos = yb * cells_per_step

            hog_feature1 = hog1[ypos:ypos+block_per_window, xpos:xpos+block_per_window].ravel()
            hog_feature2 = hog2[ypos:ypos+block_per_window, xpos:xpos+block_per_window].ravel()
            hog_feature3 = hog3[ypos:ypos+block_per_window, xpos:xpos+block_per_window].ravel()
            hog_feature = np.concatenate((hog_feature1, hog_feature2, hog_feature3))
            # Extract the image patch
            x_top_left = xpos * pix_per_cell   # convert cell to pixel
            y_top_left = ypos * pix_per_cell
            subimg = cv2.resize(ctrans_tosearch[y_top_left:y_top_left+window_size, x_top_left:x_top_left+window_size], (64,64))

            # Get color feature
            spatial_feature = bin_spatial(subimg, size=spatial_size)
            hist_feature = color_hist(subimg, nbins=hist_bins)

            # concatenate all features
            features = np.hstack((spatial_feature, hist_feature, hog_feature)).reshape(1, -1)

            # Scale features and make a prediction
            test_features = X_scaler.transform(features)
            test_prediction = svc.predict(test_features)
            if test_prediction == 1:
                xbox_left = np.int(x_top_left * scale)
                ybox_left = np.int(y_top_left * scale)
                window = np.int(window_size * scale)
                car_windows.append(((xbox_left, ybox_left+ystart), (xbox_left+window, ybox_left+ystart+window)))
                cv2.rectangle(draw_img, (xbox_left, ybox_left+ystart), (xbox_left+window, ybox_left+ystart+window), (0,0,255), 6)
    # print(car_windows)
    return car_windows

def add_heat(heatmap, bbox_list):
    for search in bbox_list:
        for box in search:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap

def draw_labeled_bboxes(img, labels):
    labeled_array, num_features = labels
    for car_number in range(1, num_features+1):
        # Find pixels with each car_number label value
        # .nonzero(): Return the indices of the elements that are non-zero
        nonzero = (labeled_array == car_number).nonzero()
        # Identify x and y values of those pixels
        y = np.array(nonzero[0])
        x = np.array(nonzero[1])
        bbox = ((np.min(x), np.min(y)), (np.max(x), np.max(y)))
        cv2.rectangle(img, bbox[0], bbox[1], (255,0,0), 6)
    return img

def vehicle_detection_pipeline(img):
    # ystart, ystop, scale, overlap, color
    searches = [
        (380, 480, 1.0, (0, 0, 255)),  # 64x64
        (380, 550, 1.6, (0, 255, 0)),  # 101x101
        (400, 610, 3.0, (255, 0, 0)),  # 161x161
        (400, 680, 4.0, (255, 255, 0)), # 256x256
    ]

    total_boxes = []
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for ystart, ystop, scale, color in searches:
        bboxes = find_cars(img, ystart, ystop, color_space, scale, svc, X_scaler, orient,
                           pix_per_cell, cell_per_block, spatial_size, hist_bins)
        total_boxes.append(bboxes)

    if len(frames) == 0:
        all_frames_heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
    else:
        all_frames_heatmap = frames[-1]

    current__frame_heat = np.zeros_like(img[:,:,0]).astype(np.float)
    if len(total_boxes) > 0:
        current_heatmap = add_heat(current__frame_heat, total_boxes)

    if len(frames) == 3:
        all_frames_heatmap -= frames[0] * 0.3**5

    all_frames_heatmap = all_frames_heatmap*0.8 + current__frame_heat

    frames.append(all_frames_heatmap)
    # Apply threshold to help remove false positives
    heat = apply_threshold(all_frames_heatmap, len(frames))
    # np.clip : Clip (limit) the values in an array.
    #           Given an interval, values outside the interval are clipped to the interval edges.
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    draw = draw_labeled_bboxes(img, labels)
    # convert BGR to RGB image
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    return draw

def lane_finding_pipeline(image):
    # undist = cv2.undistort(image, mtx, dist, None, mtx)
    binary_img= binary_pipeline(image, l_thresh=200, sx_thresh=(30, 100), hsv_thresh=([10,100,100],[30,255,255]))
    warped, M, Minv = perspective_transform(binary_img)   # May not need M, need to check
    left_fitx, right_fitx, ploty = line.find_lines(warped)
    #leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped)
    #left_fitx, right_fitx, ploty = fit_poly(image.shape, leftx, lefty, rightx, righty)

    # Calculate curvature radius of both lines and average them
    left_curverad, right_curverad = line.measure_curvature_real(warped, left_fitx, right_fitx, ploty)
    radius_of_curvature = int((left_curverad + right_curverad)/2)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    # Calculate center of the road to the center of the image
    left_x = left_fitx[-1]       # left line x position at bottom of image
    right_x = right_fitx[-1]     # right line x position at bottom of image
    offset = (1280/2) - (left_x + right_x)/2
    offset_direction = "right" if offset > 0 else "left"
    offset_meter = abs(offset * 3.7/700)

    # write radius and offset onto image
    result = cv2.putText(result, "Radius of Curvature = " + str(radius_of_curvature) + "(m)",
             (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    result = cv2.putText(result, 'Vehicle is %.2fm %s of center' %(offset_meter,offset_direction),
             (70, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255),3)

    return result

def combined_pipeline(img):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    result = vehicle_detection_pipeline(undist)
    combined = lane_finding_pipeline(result)
    return combined
    
# load a pe-trained svc model from a serialized (pickle) file
dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )

# get attributes of our svc object
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
color_space = dist_pickle["color_space"]
# ystart, ystop, scale, overlap, color

from find_lanelines import *
# for undistort image
wide_dist_pickle = pickle.load( open("wide_dist_pickle.p", "rb" ) )
mtx = wide_dist_pickle["mtx"]
dist = wide_dist_pickle["dist"]
line = Find_line()

frames = deque([], 3)

'''
image = mpimg.imread('test_images/scene00561.jpg')
plt.imshow(vehicle_detection_pipeline(image))
plt.show()
'''
# NOTE: f1_image function expects color images!!
outfile = 'combined_result.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(combined_pipeline)
white_clip.write_videofile(outfile, audio=False)
