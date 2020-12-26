"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
"""
import argparse
import datetime
import os
import sys
import timeit
import warnings

import json
import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble
import numpy as np
import pymia.data.conversion as conversion
import pymia.evaluation.writer as writer

sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))  # append the MIALab root directory to Python path
# fixes the ModuleNotFoundError when executing main.py in the console after code changes (e.g. git pull)
# somehow pip install does not keep track of packages

import mialab.data.structure as structure
import mialab.utilities.file_access_utilities as futil
import mialab.utilities.pipeline_utilities as putil

LOADING_KEYS = [structure.BrainImageTypes.T1w,
                structure.BrainImageTypes.T2w,
                structure.BrainImageTypes.GroundTruth,
                structure.BrainImageTypes.BrainMask,
                structure.BrainImageTypes.RegistrationTransform]  # the list of data we will load


def main(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str, tmp_result_dir: str):
    """Brain tissue segmentation using decision forests.

    The main routine executes the medical image analysis pipeline:

        - Image loading
        - Registration
        - Pre-processing
        - Feature extraction
        - Decision forest classifier model building
        - Segmentation using the decision forest classifier model on unseen images
        - Post-processing of the segmentation
        - Evaluation of the segmentation
    """

    # load atlas images
    putil.load_atlas_images(data_atlas_dir)

    # print('-' * 5, 'Training...')
    #
    # # crawl the training image directories
    # crawler = futil.FileSystemDataCrawler(data_train_dir,
    #                                       LOADING_KEYS,
    #                                       futil.BrainImageFilePathGenerator(),
    #                                       futil.DataDirectoryFilter())
    pre_process_params = {'skullstrip_pre': True,
                          'normalization_pre': True,
                          'registration_pre': True,
                          'coordinates_feature': True,
                          'intensity_feature': True,
                          'gradient_intensity_feature': True}

    #
    # # load images for training and pre-process
    # images = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)
    #
    # # generate feature matrix and label vector
    # data_train = np.concatenate([img.feature_matrix[0] for img in images])
    # labels_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()
    #
    # warnings.warn('Random forest parameters not properly set.')
    # forest = sk_ensemble.RandomForestClassifier(max_features=images[0].feature_matrix[0].shape[1],
    #                                             n_estimators=10,
    #                                             max_depth=10)
    #
    # start_time = timeit.default_timer()
    # forest.fit(data_train, labels_train)
    # print(' Time elapsed:', timeit.default_timer() - start_time, 's')
    #

    #
    # print('-' * 5, 'Testing...')
    #
    # # initialize evaluator
    # evaluator = putil.init_evaluator()
    #
    # # crawl the training image directories
    # crawler = futil.FileSystemDataCrawler(data_test_dir,
    #                                       LOADING_KEYS,
    #                                       futil.BrainImageFilePathGenerator(),
    #                                       futil.DataDirectoryFilter())
    #
    # # load images for testing and pre-process
    # pre_process_params['training'] = False
    # images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)
    #
    # images_prediction = []
    # images_probabilities = []
    #
    # for img in images_test:
    #     print('-' * 10, 'Testing', img.id_)
    #
    #     start_time = timeit.default_timer()
    #     predictions = forest.predict(img.feature_matrix[0])
    #     probabilities = forest.predict_proba(img.feature_matrix[0])
    #     print(' Time elapsed:', timeit.default_timer() - start_time, 's')
    #
    #     # convert prediction and probabilities back to SimpleITK images
    #     image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.uint8),
    #                                                                     img.image_properties)
    #     image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)
    #
    #     # evaluate segmentation without post-processing
    #     evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)
    #
    #     images_prediction.append(image_prediction)
    #     images_probabilities.append(image_probabilities)

    # initialize evaluator
    # create a result directory with timestamp
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-Search1')
    result_dir = os.path.join(result_dir, t)
    os.makedirs(result_dir, exist_ok=True)

    evaluator = putil.init_evaluator()

    # crawl the test image directories
    crawler = futil.FileSystemDataCrawler(data_test_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())

    # load necessary data to perform post processing
    pre_process_params['training'] = False
    images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=True) # False

    # load the prediction of the test images (segmented image
    images_prediction, images_probabilities = putil.load_prediction_images(images_test,
                                                                           tmp_result_dir,
                                                                           'tmp_seg_prob')
    # 2020-11-30-09-56-49

    # evaluate images without post-processing
    for i, img in enumerate(images_test):
        evaluator.evaluate(images_prediction[i], img.images[structure.BrainImageTypes.GroundTruth], img.id_)

    # save results without post-processing
    name = 'no_PP'
    sub_dir = os.path.join(result_dir, name)
    os.makedirs(sub_dir, exist_ok=True)

    result_file = os.path.join(sub_dir, 'results.csv')
    writer.CSVWriter(result_file).write(evaluator.results)

    # report also mean and standard deviation among all subjects
    result_summary_file = os.path.join(sub_dir, 'results_summary.csv')
    functions = {'MEAN': np.mean, 'STD': np.std}
    writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator.results)

    # clear results such that the evaluator is ready for the next evaluation
    evaluator.clear()

    # define parameters for grid search
    post_process_param_list = []
    # gauss parameters

    '''gauss_temp = np.zeros((3, 10))  # only change n to get more parameters for testing
    gauss = np.arange(10)
    for i in (1, 3):
        gauss_temp[:i] = gauss

    gauss_dims = gauss_temp.transpose()'''

    gauss_dims = np.arange(1.0, 3.0, 1.0)

    # bilateral parameters
    '''bil_temp = np.zeros((3, 10))    # only change n to get more parameters for testing
    bil = np.arange(10)
    for i in (1, 3):
        bil_temp[:i] = bil

    bil_dims = bil_temp.transpose()'''
    bil_dims = np.arange(1.0, 14.0, 2.0)

    # schan parameters
    '''schan_temp = np.zeros((2, 10))      # only change n to get more parameters for testing
    schan_array = np.arange(10)
    for i in (1, 2):
        schan_temp[:i] = schan_array

    schan = schan_temp.transpose()'''
    schan = np.arange(1.0, 14.0, 2.0)

    for g_d in gauss_dims:
        for b_d in bil_dims:
            for sn in schan:
                post_process_param_list.append({'crf_post': bool(True),
                                                'gauss_dims': g_d,
                                                'bil_dims': b_d,
                                                'schan' : sn})

    #gridsearch = 1.0 # for easier evaluation

    #post_process_params = {'crf_post': True}
    for post_process_params in post_process_param_list:

        name = 'PP-GD-' + str(post_process_params.get('gauss_dims')).replace('.','_') +\
               '-BD-' + str(post_process_params.get('bil_dims')).replace('.','_') +\
               '-S-' + str(post_process_params.get('schan')).replace('.','_')

        #gridsearch = gridsearch + 1

        sub_dir = os.path.join(result_dir, name)
        os.makedirs(sub_dir, exist_ok=True)

        # write the used parameter into a text file and store it in the result folder
        completeName = os.path.join(sub_dir, "parameter.txt")
        file1 = open(completeName, "w+")
        json.dump(post_process_params, file1)
        file1.close()

        # post-process segmentation and evaluate with post-processing
        images_post_processed = putil.post_process_batch(images_test, images_prediction, images_probabilities,
                                                         post_process_params, multi_process=True) # False


        for i, img in enumerate(images_test):
            evaluator.evaluate(images_post_processed[i], img.images[structure.BrainImageTypes.GroundTruth],
                               img.id_ + '-PP')

            # save results
            #sitk.WriteImage(images_prediction[i], os.path.join(result_dir, images_test[i].id_ + '_SEG.mha'), True)
            sitk.WriteImage(images_post_processed[i], os.path.join(sub_dir, images_test[i].id_ + '_SEG-PP.mha'), True)


        # use two writers to report the results
        os.makedirs(result_dir, exist_ok=True)  # generate result directory, if it does not exists
        result_file = os.path.join(sub_dir, 'results.csv')
        writer.CSVWriter(result_file).write(evaluator.results)

        print('\nSubject-wise results...')
        writer.ConsoleWriter().write(evaluator.results)

        # report also mean and standard deviation among all subjects
        result_summary_file = os.path.join(sub_dir, 'results_summary.csv')
        functions = {'MEAN': np.mean, 'STD': np.std}
        writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator.results)
        print('\nAggregated statistic results...')
        writer.ConsoleStatisticsWriter(functions=functions).write(evaluator.results)

        # clear results such that the evaluator is ready for the next evaluation
        evaluator.clear()


if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-result')),
        help='Directory for results.'
    )

    parser.add_argument(
        '--tmp_result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/tmp_result_dir')),
        help='Directory to store segmented data prior to post processing.'
    )

    parser.add_argument(
        '--data_atlas_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/atlas')),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_train_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/train/')),
        help='Directory with training data.'
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/test/')),
        help='Directory with testing data.'
    )

    args = parser.parse_args()
    main(args.result_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir, args.tmp_result_dir)
