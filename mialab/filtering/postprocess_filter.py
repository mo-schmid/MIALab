import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy import ndimage


def smooth_probabilities(image: sitk.Image, variance=1.0) -> sitk.Image:
    """ Gaussian smoothing of the probability image

    Args:
        image (sitk.Image): vector image with probabilities for each pixel and label
        variance (Float): Variance of the gaussian smoothing

    Returns (sitk.Image): smoothed vector image

    """
    # get number of labels
    n_labels = image.GetNumberOfComponentsPerPixel()

    # generate filter to extract probabilities of each label
    extract_label_filter = sitk.VectorIndexSelectionCastImageFilter()

    # generate gaussian filter to smooth the probability images
    gauss_filter = sitk.DiscreteGaussianImageFilter()
    gauss_filter.SetVariance(variance)

    # generate vector to store the different label probabilities image
    images_probabilities = []

    # extract the probabilities for a single label and smooth the "image"
    for label in range(n_labels):
        probabilities = extract_label_filter.Execute(image, label, sitk.sitkFloat32)
        images_probabilities.append(gauss_filter.Execute(probabilities))

    # compose the single images back to a vector image
    compose_filter = sitk.ComposeImageFilter()
    img_out = compose_filter.Execute(images_probabilities)

    # arr_image = sitk.GetArrayFromImage(img_out)
    # plt.imshow(arr_image[:, :, 90], cmap='jet')
    # plt.show()


    return img_out


def get_largest_segment(image: sitk.Image) -> sitk.Image:
    # get number of labels
    img_statistic = sitk.StatisticsImageFilter()
    img_statistic.Execute(image)

    min_val = int(img_statistic.GetMinimum())
    max_val = int(img_statistic.GetMaximum())

    # create empty output image
    img_out = sitk.Image(image.GetSize(), image.GetPixelIDValue())
    img_out.CopyInformation(image)

    # setup connected components filter
    connected_comp_filter = sitk.ConnectedComponentImageFilter()
    connected_comp_filter.FullyConnectedOn()

    # extract largest segment
    for label in range(min_val + 1, max_val + 1):
        img_label = image == label
        seg = connected_comp_filter.Execute(img_label != 0)
        seg = (sitk.RelabelComponent(seg) == 1) * label
        img_out = img_out + seg

    # arr_image = sitk.GetArrayFromImage(img_out)
    # plt.imshow(arr_image[:, :, 90], cmap='jet')
    # plt.show()

    return img_out


def fill_keyhole_probabilistic(image: sitk.Image, image_prob: sitk.Image) -> sitk.Image:

    arr_image = sitk.GetArrayFromImage(image)
    arr_prob = sitk.GetArrayFromImage(image_prob)

    label_prob = np.argmax(arr_prob, axis=3)

    arr_image[arr_image == 0] = label_prob[arr_image == 0]



    img_out = sitk.GetImageFromArray(arr_image)
    img_out.CopyInformation(image)



    # img_out = sitk.GetImageFromArray(label_prob)
    # img_out.CopyInformation(image)



    # ##-------------------------------------------------------------------------------------------------------------------
    # ## only for in function plotting
    # ##-------------------------------------------------------------------------------------------------------------------
    # layer = 60
    #
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    #
    # ax1.imshow(sitk.GetArrayFromImage(image)[layer,:,:], cmap='jet')
    #
    # tmp_image = sitk.GetArrayFromImage(img_out)
    # ax2.imshow(tmp_image[layer, :, :], cmap='jet')
    #
    # ax3.imshow(label_prob[layer, :, :], cmap='jet')
    # plt.show()


    return img_out


def binary_fill_keyhole(image: sitk.Image) -> sitk.Image:
    # convert sitk image into numpy array
    arr_image = sitk.GetArrayFromImage(image)
    arr_out = np.zeros(arr_image.shape)

    # define the neighbourhood (filter kernel)
    kernel = np.ones((3, 3, 3))
    region_threshold = 17

    labels = np.arange(1, arr_image.max() + 1)

    for label in labels:
        arr_binary = (arr_image == label).astype('int8')
        # plt.imshow(arr_binary[68,:,:], cmap='gray')
        # plt.show()

        arr_conv = ndimage.convolve(arr_binary, kernel, mode='constant', cval=0)

        arr_tmp = (arr_binary == 0) & (arr_conv > region_threshold)
        # arr_tmp = (arr_tmp + arr_binary.astype('bool')) * label
        # arr_out = arr_out + arr_tmp
        arr_image[arr_tmp] = label

    # image_out = sitk.GetImageFromArray(arr_out)
    img_out = sitk.GetImageFromArray(arr_image)
    img_out.CopyInformation(image)

    return img_out


if __name__ == "__main__":

    # allows to run only the post processing of an image

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

    # load the label images
    images_prediction = []
    images_probabilities = []
    image_ids = []

    for filename in os.listdir(os.path.join(source_dir, dataset)):
        if filename.endswith("_SEG.mha"):
            images_prediction.append(sitk.ReadImage(os.path.join(source_dir, dataset, filename)))
            image_ids.append(filename.replace('_SEG.mha', ''))
        elif filename.endswith("_PROB.mha"):
            images_probabilities.append(sitk.ReadImage(os.path.join(source_dir, dataset, filename)))

    # =======================================================================================================================

    # post-process the segments
    images_pp = []
    for i, id in enumerate(image_ids):
        # images_pp.append(binary_closing(images_prediction[i]))
        # images_pp.append(binary_fill_keyhole(images_prediction[i]))
        # images_pp.append(connected_segments(images_prediction[i]))
        # images_pp.append(smooth_probabilities(images_probabilities[i]))
        images_pp.append(fill_keyhole_probabilistic(get_largest_segment(images_prediction[i]),smooth_probabilities(images_probabilities[i], variance=1)))
        # images_pp.append(binary_fill_keyhole(connected_segments(images_prediction[i])))

    # =======================================================================================================================

    # save post-processed images
    if (save_images):
        t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        result_dir = os.path.join(result_dir, t)
        os.makedirs(result_dir, exist_ok=True)

        completeName = os.path.join(result_dir, t + ".txt")
        file1 = open(completeName, "w+")
        file1.write("post processing based on result: " + str(dataset))
        file1.write("\nsegments with highest probability based on a smoothed probability image with variance = 9    ")
        file1.close()

        for i, id in enumerate(image_ids):
            sitk.WriteImage(images_pp[i], os.path.join(result_dir, id + '_PP.mha'), True)
