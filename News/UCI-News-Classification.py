# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import logging
import sys

from optparse import OptionParser
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="Number of features when using the hashing vectorizer.")
# TODO Options for feature inclusion (i.e. readability score(s), other biases)


def filter_data(text_array):
    """Filters text data."""
    filtered_texts = []
    spec_char_re = re.compile('[^a-z0-9 ]')

    for text in text_array:
        # Make lowercase
        text = text.lower()
        # Replace special chars with spaces
        text = spec_char_re.sub(' ', text)
        # Remove duplicate spaces
        text = ' '.join(text.split())

        filtered_texts.append(text)

    return filtered_texts


def enumerate_targets(targets):
    # Sorted for consistency
    unique = sorted(np.unique(targets))
    return [unique.index(target) for target in targets], unique


def extract_features(X_train, X_test, y_train, Y_test, use_hashing,
                     chi, n_features):
    """
        Args:
            X_train:     Training data features
            X_test:      Testing data features
            y_train:     Training data targets
            use_hashing: Use hashing vectorizer instead of tf-idf
            chi:         Use chi-squared feature selection
            n_features:  The number of features to use (hashing only)
        Returns:
            X_train, X_test, feature_names
    """
    print('Extracting features from text via {} transformation.'.format(
        'hashing' if use_hashing else 'TF-IDF'))

    # Create vectorizer and transform training data
    if use_hashing:
        vectorizer = HashingVectorizer(stop_words='english',
                                       non_negative=True,
                                       n_features=n_features)
        X_train = vectorizer.transform(data_train.data)
    else:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                     stop_words='english')
        X_train = vectorizer.fit_transform(data_train.data)

    # Transform testing data
    X_test = vectorizer.transform(data_test.data)

    print('[Train] n_samples: %d, n_features: %d'.format(*X_train.shape))
    print('[Test]  n_samples: %d, n_features: %d'.format(*X_test.shape))

    # mapping from integer feature name to original token string
    if opts.use_hashing:
        feature_names = None
    else:
        feature_names = vectorizer.get_feature_names()

    if select_chi2:
        print('Extracting %d best features using a chi-squared test'.format(
                opts.select_chi2))
        ch2 = SelectKBest(chi2, k=select_chi2)
        X_train = ch2.fit_transform(X_train, y_train)
        X_test = ch2.transform(X_test)
        if feature_names:
            feature_names = [feature_names[i] for i
                             in ch2.get_support(indices=True)]

    if feature_names:
        feature_names = np.asarray(feature_names)

    return X_train, X_test, feature_names


def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate(target_names):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
        print()

    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))

    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


opts, args = op.parse_args()
if len(args) > 0:
    op.error('Script received non-applicable args: {}.'.format(
        ' '.join(args)))
    sys.exit(1)


# TODO Load all data
all_data, all_targets = ...

# Filter text data
all_data_filtered = filter_data(all_data)

# Enumerate targets
all_targets_enum, target_names = enumerate_targets(all_targets)

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(
        all_data_filtered, all_targets_enum, test_size=0.67)

# ******************* MODEL FITTING *********************

results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=100), "Random forest")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                            dual=False, tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty)))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))

print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
  ('classification', LinearSVC())
])))

# ********************* PLOTTING **********************
indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()
