# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import logging
import sys
import re
import string

from optparse import OptionParser
from time import time

from textstat.textstat import textstat as ts

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
op.add_option('--chi2_select',
              action='store', type=int, dest='select_chi2',
              help='Select some number of features using a chi-squared test')
op.add_option('--full_report',
              action='store_true', dest='all_details',
              help='Print a confusion matrix and classification report.')
op.add_option('--top10',
              action='store_true', dest='print_top10',
              help='Print ten most discriminative terms per class '
                   'for applicable classifiers.')
op.add_option('--use_hashing',
              action='store_true',
              help='Use a hashing vectorizer instead of TF-IDF.')
op.add_option('--n_features',
              action='store', type=int, default=2 ** 16,
              help='Number of features when using the hashing vectorizer.')
op.add_option('--n_publishers',
              action='store', type=int, default=20,
              help='The top number of publishers to train on.')


def filter_data(text_array):
    """Filters text data."""
    filtered_texts = []
    spec_char_re = re.compile('[^\x00-\x7F]+')
    punc_char_re = re.compile('[^a-z0-9 ]+')

    for text in text_array:
        # Make lowercase
        text = text.lower()
        # Replace special chars (unicode) with spaces
        text = spec_char_re.sub(' ', text)
        # Remove duplicate spaces (and \r\t\n)
        text = ' '.join(text.split())
        # Remove punctuation
        text = punc_char_re.sub('', text)

        filtered_texts.append(text)

    return filtered_texts


def enumerate_targets(targets):
    # Sorted for consistency
    unique = sorted(np.unique(targets))
    return [unique.index(target) for target in targets], unique


def extract_features(X_train, X_test, y_train, use_hashing,
                     chi, n_features):
    """Extracts features from text.

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
        X_train = vectorizer.transform(X_train)
    else:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                     stop_words='english')
        X_train = vectorizer.fit_transform(X_train)

    # Transform testing data
    X_test = vectorizer.transform(X_test)

    print('[Train] n_samples: {}, n_features: {}'.format(*X_train.shape))
    print('[Test]  n_samples: {}, n_features: {}'.format(*X_test.shape))

    # mapping from integer feature name to original token string
    if opts.use_hashing:
        feature_names = None
    else:
        feature_names = vectorizer.get_feature_names()

    if chi:
        print('Extracting {} best features using a chi-squared test'.format(
                opts.select_chi2))
        ch2 = SelectKBest(chi2, k=chi)
        X_train = ch2.fit_transform(X_train, y_train)
        X_test = ch2.transform(X_test)
        if feature_names:
            feature_names = [feature_names[i] for i
                             in ch2.get_support(indices=True)]

    if feature_names:
        feature_names = np.asarray(feature_names)

    return X_train, X_test, feature_names


def benchmark(clf, X_train, y_train, X_test, y_test):
    print('_' * 80)
    print('Training:')
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print('Train time:    %0.3fs' % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print('Test time:     %0.3fs' % test_time)

    accuracy = metrics.accuracy_score(y_test, pred)
    print('Accuracy:      %0.3f' % accuracy)
    kappa = metrics.cohen_kappa_score(y_test, pred)
    print('Cohen\'s Kappa: %0.3f' % kappa)

    if hasattr(clf, 'coef_'):
        print('Dimensionality: %d' % clf.coef_.shape[1])
        print('Density:        %f' % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print('Top 10 keywords per class:')
            for i, label in enumerate(target_names):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print('%s: %s' % (label, ' '.join(feature_names[top10])))
        print()

    if opts.all_details:
        print('Classification Report:')
        print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))
        print('Confusion Matrix:')
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]

    return clf_descr, accuracy, kappa, train_time, test_time


opts, args = op.parse_args()
if len(args) > 0:
    op.error('Script received non-applicable args: {}.'.format(
        ' '.join(args)))
    sys.exit(1)


# load all data
print('Reading in data.')
df = pd.read_csv('uci-news-inchunks.csv', dtype=object, keep_default_na=False)

# Top 10
target_subset = df.PUBLISHER.value_counts().index.values[:opts.n_publishers]

# Sample data
if False:
    print('Sampling data.')
    df = df.sample(frac=0.1)
else:
    print('Extracting subset of news sources.')
    df = df.loc[df.PUBLISHER.isin(target_subset)]

all_data    = df.TEXT.values
all_targets = df.PUBLISHER.values

# Filter text data
print('Filtering text data.')
all_data_filtered = filter_data(all_data)

# Enumerate targets
print('Enumerating targets.')
all_targets_enum, target_names = enumerate_targets(all_targets)

print('{} targets: {}'.format(len(target_names), target_names))


# Split data into train/test
print('Splitting data into train/test partitions.')
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        all_data_filtered, all_targets_enum, test_size=0.67)

# Extract features
X_train, X_test, feature_names = extract_features(
        X_train_raw, X_test_raw, y_train, use_hashing=opts.use_hashing,
        chi=opts.select_chi2, n_features=opts.n_features)

bm_data = (X_train, y_train, X_test, y_test)
# ******************* MODEL FITTING *********************

results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver='lsqr'), 'Ridge Classifier'),
        (Perceptron(n_iter=50), 'Perceptron'),
        (PassiveAggressiveClassifier(n_iter=50), 'Passive-Aggressive'),
        (KNeighborsClassifier(n_neighbors=10), 'kNN'),
        (RandomForestClassifier(n_estimators=100), 'Random forest')):
    print('_' * 80)
    print(name)
    results.append(benchmark(clf, *bm_data))

for penalty in ['l2', 'l1']:
    print('_' * 80)
    print('%s penalty' % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                            dual=False, tol=1e-3),
                             *bm_data))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty),
                             *bm_data))

# Train SGD with Elastic Net penalty
print('_' * 80)
print('Elastic-Net penalty')
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty='elasticnet'),
                         *bm_data))

# Train NearestCentroid without threshold
print('_' * 80)
print('NearestCentroid (aka Rocchio classifier)')
results.append(benchmark(NearestCentroid(), *bm_data))

# Train sparse Naive Bayes classifiers
print('_' * 80)
print('Naive Bayes')
results.append(benchmark(MultinomialNB(alpha=.01), *bm_data))
results.append(benchmark(BernoulliNB(alpha=.01), *bm_data))

print('_' * 80)
print('LinearSVC with L1-based feature selection')
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', LinearSVC(penalty='l1', dual=False, tol=1e-3)),
  ('classification', LinearSVC())
]), *bm_data))


"""# Readability
flesch_reading_score  = ts.flesch_reading_ease(article_txt)
smog_score            = ts.smog_index(article_txt)
flesch_kincaid_score  = ts.flesch_kincaid_grade(test_data)
coleman_liau_score    = ts.coleman_liau_index(test_data)
ari_score             = ts.automated_readability_index(test_data)
dale_chall_score      = ts.dale_chall_readability_score(test_data)
linsear_write_score   = ts.linsear_write_formula(test_data)
gunning_fog_score     = ts.gunning_fog(test_data)
readability_consensus = ts.text_standard(test_data)

print(("Flesch Reading Ease:          {}\n" +
       "Flesch Kincaid Grade:         {}\n" +
       "SMOG Index:                   {}\n" +
       "Automated Readability Index:  {}\n" +
       "Dale Chall Readability Score: {}\n" +
       "Linsear Write Score:          {}\n" +
       "Gunning Fog:                  {}\n" +
       "-" * 79                             +
       "Readability Consensus:        {}").format(
       flesch_reading_score, flesch_kincaid_grade, smog_score,
       ari_score, dale_chall_score, linsear_write_score,
       gunning_fog_score, readability_consensus))"""

# ********************* PLOTTING **********************
indices = np.arange(len(results)) * 1.3

# Sort results by Kappa
results = sorted(results, key=lambda x: x[2])
results = [[x[i] for x in results] for i in range(5)]

clf_names, accuracy, kappa, training_time, test_time = results
print('Top Classifier: {}\n'
      'Accuracy:       {}\n'
      'Cohen\'s Kappa:  {}\n'
      'Train Time:     {}\n'
      'Test Time:      {}\n'.format(
      clf_names[-1], accuracy[-1], kappa[-1],
      training_time[-1], test_time[-1]))
# Normalize times
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12 * 1.3, 8))
plt.title('Classifier Scores')
plt.barh(indices,      accuracy,      .2, label='accuracy',
         color='navy')
plt.barh(indices + .3, kappa,         .2, label='kappa',
         color='g')
plt.barh(indices + .6, training_time, .2, label='training time',
         color='c')
plt.barh(indices + .9, test_time,     .2, label='test time',
         color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()
