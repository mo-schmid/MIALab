

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy import ndimage




def binary_closing(image: sitk.Image) -> sitk.Image:


    return image


def binary_fill_keyhole(image: sitk.Image) -> sitk.Image:

    arr_image = sitk.GetArrayFromImage(image)
    arr_out = np.zeros(arr_image.shape)

    kernel = np.ones((3, 3, 3))
    region_threshold = 17

    labels = np.arange(1, arr_image.max()+1)

    for label in labels:

        arr_binary = (arr_image == label).astype('int8')
        # plt.imshow(arr_binary[68,:,:], cmap='gray')
        # plt.show()

        arr_conv = ndimage.convolve(arr_binary, kernel, mode='constant',cval=0)

        arr_tmp = (arr_binary == 0) & (arr_conv > region_threshold)
        # arr_tmp = (arr_tmp + arr_binary.astype('bool')) * label
        # arr_out = arr_out + arr_tmp
        arr_image[arr_tmp] = label

    # image_out = sitk.GetImageFromArray(arr_out)
    image_out = sitk.GetImageFromArray(arr_image)
    image_out.CopyInformation(image)


    return image_out













if __name__ == "__main__":

    #allows to run only the post processing of an image

    import os
    import sys
    import datetime
    import mialab.utilities.pipeline_utilities as putil
    import mialab.utilities.file_access_utilities as futil
    import mialab.data.structure as structure
    from pathlib import Path

    # build used directories
    script_dir = os.path.dirname(sys.argv[0])
    root_dir = Path(script_dir).resolve().parents[1]
    source_dir = os.path.normpath(os.path.join(root_dir, './data/tmp_results'))
    result_dir = os.path.normpath(os.path.join(root_dir, './bin/results_pp'))

    # string representing the folder where the segmented data is stored
    dataset = '2020-10-30-18-31-15'

    # save_images = False
    save_images = True

    #load the label images
    images_prediction = []
    images_probabilities = []
    image_ids = []

    for filename in os.listdir(os.path.join(source_dir, dataset)):
        if filename.endswith("_SEG.mha"):
            images_prediction.append(sitk.ReadImage(os.path.join(source_dir, dataset, filename)))
            image_ids.append(filename.replace('_SEG.mha',''))
        elif filename.endswith("_PROB.mha"):
            images_probabilities.append(sitk.ReadImage(os.path.join(source_dir, dataset, filename)))


    # post-process the segments
    images_pp = []
    for i, id in enumerate(image_ids):
        # images_pp.append(binary_closing(images_prediction[i]))
        images_pp.append(binary_fill_keyhole(images_prediction[i]))


    # save post-processed images
    if (save_images):
        t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        result_dir = os.path.join(result_dir, t)
        os.makedirs(result_dir, exist_ok=True)

        completeName = os.path.join(result_dir, t + ".txt")
        file1 = open(completeName, "w+")
        file1.write("post processing based on result: " + str(dataset))
        file1.close()

        for i, id in enumerate(image_ids):
            sitk.WriteImage(images_pp[i], os.path.join(result_dir, id + '_PP.mha'), True)











