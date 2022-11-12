import cv2
import os
import time
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



NB_IMAGES = 1000
SIMILARITY_THRESHOLD = 800


# TODO find good threshold
# TODO possibly use multiple threads https://towardsdatascience.com/which-is-faster-python-threads-or-processes-some-insightful-examples-26074e90848f
def main():
    env_name = 'Seaquest-v5'
    image_folder = 'images'

    images = os.listdir(image_folder + '/' + env_name)[:NB_IMAGES]
    # print(images)

    differences = []

    lst = [cv2.imread(image_folder + '/' + env_name + '/' + images[0]) / 255]

    start = time.time()

    for i in tqdm(range(1, len(images))):
        img = cv2.imread(image_folder + '/' + env_name + '/' + images[i]) / 255

        diff = min_difference_from_list(lst, img)

        differences.append(diff)

        if diff > SIMILARITY_THRESHOLD:
            lst.append(img)


    stop = time.time()

    print(f"Time: {int(stop - start)}s")

    print(f"Images choosen: {len(lst)} (out of {NB_IMAGES})")

    plt.hist(differences, bins=20, density=True)
    plt.figure()
    plt.plot(range(len(differences)), differences)
    plt.show()



def test_functions():



    # Divide by 255, otherwise images are uint8 and subtraction will go in overflow (e.g. 1 - 3 == 254)
    img0 = cv2.imread(image_folder + '/' + env_name + '/' + images[0]) / 255
    img1 = cv2.imread(image_folder + '/' + env_name + '/' + images[1]) / 255
    img2 = cv2.imread(image_folder + '/' + env_name + '/' + images[2]) / 255
    img3 = cv2.imread(image_folder + '/' + env_name + '/' + images[3]) / 255


    lst = [img1, img2, img3]


    # # Test difference_from_list
    # diff = np.abs(np.subtract(img0, img1))
    # print(np.sum(diff))
    # print(difference_from_list([img0], img1))
    # print(difference_from_list([img1], img0))


    # # Check whether we can speed up by averaging the list (answer: no)
    # print(difference_from_list(lst, img0))
    # print(difference_from_centroid_list(lst, img0))

    # # Better to use min rather than mean
    # print(difference_from_list([img0, img1, img2, img3], img0))
    # print(min_difference_from_list([img0, img1, img2, img3], img0))


    # cv2.imshow('img0', img0)
    # cv2.imshow('img1', img1)
    # cv2.imshow('diff', diff)
    #
    # print(np.sum(diff))
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Get the sum of pixerlwise difference between two images
def image_difference(img0, img1):
    return np.sum(np.abs(np.subtract(img0, img1)))

# Get the average difference between an image and a list of images
def difference_from_list(img_list, img):
    assert len(img_list) >= 1, 'The img list should have at least 1 image!'
    diffs = []
    for comp_img in img_list:
        diffs.append(image_difference(comp_img, img))
    return np.mean(diffs)

# Get the difference between an image and the average of a list of images
def difference_from_centroid_list(img_list, img):
    # Average the images in the list
    avg_img = np.mean(img_list, axis=0)
    return image_difference(avg_img, img)


# Get the average difference between an image and a list of images
def min_difference_from_list(img_list, img):
    assert len(img_list) >= 1, 'The img list should have at least 1 image!'
    diffs = []
    for comp_img in img_list:
        diffs.append(image_difference(comp_img, img))
    return min(diffs)


if __name__ == "__main__":
    main()
