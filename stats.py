import cv2
import numpy as np
import traceback
import matplotlib.pyplot as plt


def get_stats(image_channels, image, mask, cell_mask, cell_image_crop, cell_mask_crop, contour):

    try:

        cell_stats = {}

        cell_area = cv2.contourArea(contour)

        cell_overlap = determine_overlap(mask, cell_mask, contour)

        for i, channel in enumerate(image_channels):

            img = image[i]
            cell_img = cell_image_crop[i].copy()

            cell_brightness = int(np.mean(cell_img[cell_mask_crop != 0]))
            cell_background_brightness = int(np.mean(cell_img[cell_mask_crop == 0]))
            cell_contrast = cell_brightness / cell_background_brightness

            image_brightness = int(np.mean(img[mask != 0]))
            image_background_brightness = int(np.mean(img[mask == 0]))
            image_contrast = image_brightness / image_background_brightness

            img_laplacian = int(cv2.Laplacian(img, cv2.CV_64F).var())
            cell_laplacian = int(cv2.Laplacian(cell_img, cv2.CV_64F).var())

            cell_stats[f"Cell Laplacian [{channel}]"] = cell_laplacian
            cell_stats[f"Cell Brightness [{channel}]"] = cell_brightness
            cell_stats[f"Cell Contrast [{channel}]"] = cell_contrast

            cell_stats[f"Image Laplacian [{channel}]"] = img_laplacian
            cell_stats[f"Image Brightness [{channel}]"] = image_brightness
            cell_stats[f"Image Contrast [{channel}]"] = image_contrast

        cell_stats["Cell Area (Pixels)"] = cell_area
        cell_stats["Cell Overlap (%)"] = cell_overlap

    except:
        cell_stats = None

    return cell_stats


def determine_overlap(mask, cell_mask, contour):

    overlap_percentage = None

    try:

        overlap_mask = np.zeros_like(mask)
        overlap_mask[mask != 0] = 1
        overlap_mask[cell_mask != 0] = 0
        overlap_mask = overlap_mask.astype(np.uint8)

        overlap_cell_mask = np.zeros_like(cell_mask)
        cv2.drawContours(overlap_cell_mask, [contour], contourIdx=-1, color=(1, 1, 1), thickness=1)

        # length of contour
        cnt_pixels = len(contour)

        # dilate the contours mask. Neighbouring contours will now overlap.
        kernel = np.ones((3, 3), np.uint8)
        overlap_mask = cv2.dilate(overlap_mask, kernel, iterations=1)

        # get overlapping pixels
        overlap = cv2.bitwise_and(overlap_cell_mask,overlap_mask)

        # count the number of overlapping pixels
        overlap_pixels = len(overlap[overlap == 1])

        # calculate the overlap percentage
        overlap_percentage = int((overlap_pixels / cnt_pixels) * 100)

    except:
        print(traceback.format_exc())

    return overlap_percentage

#
# print(cached_data.keys())
#
# for stat_name in cached_data["stats"][0].keys():
#
#     print(stat_name)
#
#     plot_data = []
#
#     for label in np.unique(cached_data["labels"]):
#
#         label_indices = np.where(cached_data["labels"] == label)[0]
#
#         label_stats = np.take(cached_data["stats"], label_indices)
#
#         label_stats = [stat for stat in label_stats if stat != None]
#
#         cell_area = np.array([stat[stat_name] for stat in label_stats])
#
#         plot_data.append(cell_area)
#
#     import matplotlib.pyplot as plt
#     import matplotlib
#     # matplotlib.use('svg')
#
#     fig, ax = plt.subplots()
#     ax.boxplot(plot_data, labels = antibiotic_list, widths = 0.7)
#     ax.set_ylabel(stat_name)
#     plt.show()