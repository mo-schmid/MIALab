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


def main(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str):
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

    print('-' * 5, 'Pre-Processing...')

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_train_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())
    pre_process_params = {'skullstrip_pre': True,
                          'normalization_pre': True,
                          'registration_pre': True,
                          'coordinates_feature': True,
                          'intensity_feature': True,
                          'gradient_intensity_feature': True}

    # load images for training and pre-process
    images = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

    # generate feature matrix and label vector
    data_train = np.concatenate([img.feature_matrix[0] for img in images])
    labels_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()


    # crawl the test image directories
    crawler = futil.FileSystemDataCrawler(data_test_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())

    # load images for testing and pre-process
    pre_process_params['training'] = False
    images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

    # create a result directory with timestamp
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = os.path.join(result_dir, t)
    os.makedirs(result_dir, exist_ok=True)


    # set parameter for grid search gridsearch
    search_tree_estimators = np.array([10, 15, 20, 25])
    search_tree_max_depth = np.array([10, 12, 14, 16, 18])

    start_time_gridsearch = timeit.default_timer()

    # execute the grid search
    for estimator in search_tree_estimators:
        for depth in search_tree_max_depth:

            print('-' * 5, f"Training: n estimator: {estimator}, max depth{depth}", '-' * 5)

            # warnings.warn('Random forest parameters not properly set.')
            forest = sk_ensemble.RandomForestClassifier(max_features=images[0].feature_matrix[0].shape[1],
                                                        n_estimators=estimator,
                                                        max_depth=depth)

            start_time = timeit.default_timer()
            forest.fit(data_train, labels_train)
            print(' Time elapsed:', timeit.default_timer() - start_time, 's')

            # create sub-directory for results
            name = 'estim-' + str(estimator).replace('.', '_') + \
                   '-depth-' + str(depth).replace('.', '_')
            sub_dir = os.path.join(result_dir, name)
            os.makedirs(sub_dir, exist_ok=True)

            # store used parameter in text file
            completeName = os.path.join(sub_dir, "parameter.txt")
            file1 = open(completeName, "w+")
            json.dump({'tree_estimator': int(estimator), 'max_depth': int(depth)}, file1)
            file1.close()

            print('-' * 5, 'Testing...')

            # initialize evaluator
            evaluator = putil.init_evaluator()

            images_prediction = []
            images_probabilities = []

            for img in images_test:
                print('-' * 10, 'Testing', img.id_)

                start_time = timeit.default_timer()
                predictions = forest.predict(img.feature_matrix[0])
                probabilities = forest.predict_proba(img.feature_matrix[0])
                print(' Time elapsed:', timeit.default_timer() - start_time, 's')

                # convert prediction and probabilities back to SimpleITK images
                image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.uint8),
                                                                                img.image_properties)
                image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)

                # evaluate segmentation without post-processing
                evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

                images_prediction.append(image_prediction)
                images_probabilities.append(image_probabilities)

            # save results
            for i, img in enumerate(images_test):
                sitk.WriteImage(images_prediction[i], os.path.join(sub_dir, images_test[i].id_ + '_SEG.mha'), True)


            # use two writers to report the results
            os.makedirs(sub_dir, exist_ok=True)  # generate result directory, if it does not exists
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

    print('Total Time elapsed:', timeit.default_timer() - start_time_gridsearch, 's')

if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-result/gridsearch_randomForest')),
        help='Directory for results.'
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
    main(args.result_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir)
