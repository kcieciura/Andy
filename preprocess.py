import numpy as np
import operator

from skimage import measure
from skimage import transform
from matplotlib import pyplot as plt

IOU_THRESHOLD = 0.1
CONTOUR_LEVEL = 0.8


def __get_contour_box(contour):
    x1, y1 = np.min(contour, axis=0)  # get min of each column, gets min x and min y (upper leftmost corner)
    x2, y2 = np.max(contour, axis=0)  # max x and max y(lower rightmost corner)
    return x1, y1, x2, y2


# remove overlapping contours such as the holes in 6,8,9
def __remove_overlap_contours(contours):
    new_contours = []
    indices_to_remove = []

    for i, contour1 in enumerate(contours):

        for j, contour2 in enumerate(contours):

            if contour1 is not contour2:

                box1 = __get_contour_box(contour1)
                box2 = __get_contour_box(contour2)
                iou, box1_area, box2_area = __get_iou(box1, box2)

                # remove smaller contour
                if iou > IOU_THRESHOLD:

                    if box1_area < box2_area and i not in indices_to_remove:

                        indices_to_remove.append(i)
                    elif box2_area >= box1_area and j not in indices_to_remove:

                        indices_to_remove.append(j)

    for i in range(len(contours)):

        if i not in indices_to_remove:
            new_contours.append(contours[i])

    return new_contours


def __get_iou(box1, box2):
    xi1 = max(box1[0], box2[0])  # x1
    yi1 = max(box1[1], box2[1])  # y1
    xi2 = min(box1[2], box2[2])  # x2
    yi2 = min(box1[3], box2[3])  # y2

    inter_width = max((xi2 - xi1), 0)
    inter_height = max(yi2 - yi1, 0)
    inter_area = inter_width * inter_height

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # compute the IoU
    iou = inter_area / union_area
    return iou, box1_area, box2_area


def getChars(pixels):
    # use sci-kit image to find the contours of the image
    contours = measure.find_contours(pixels, CONTOUR_LEVEL)

    # remove overlapping contours
    contours = __remove_overlap_contours(contours)

    # create a dictionary of contour box coordinates and x1 values as keys (upper leftmost corner)
    resized_char_dict = dict()

    # create dictionary of key -> x1 coordinate and value -> contour
    for i, contour in enumerate(contours):
        x1, y1, x2, y2 = __get_contour_box(contour)
        char_image = transform.resize(pixels[int(x1):int(x2), int(y1):int(y2)], (32, 32))
        resized_char_dict[x1] = char_image

    extracted_chars = []

    # sort the contours based on x1
    for key in sorted(resized_char_dict.keys()):
        extracted_chars.append(resized_char_dict[key])

    # convert to numpy so we can feed it to the model
    extracted_chars = np.array(extracted_chars)
    return extracted_chars


def __displayContours(contours, pixels):
    print('Num of contours: ' + str(len(contours)))
    print(pixels.shape)

    # convert data from (x,y) to (row,col)
    pixels = pixels.transpose()

    fig, ax = plt.subplots()
    ax.imshow(pixels, cmap=plt.cm.gray)

    for contour in contours:
        ax.plot(contour[:, 0], contour[:, 1], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


