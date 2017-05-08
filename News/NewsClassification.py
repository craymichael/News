"""BSD 3-Clause License

  Copyright (c) 2017, Zach
  All rights reserved. 

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met: 

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
  * Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import math
import operator
import logging
import sys
import re
import string

from optparse import OptionParser
from time import time
from collections import Counter

from textstat.textstat import textstat as ts
from gender_detector.gender_detector import GenderDetector

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
op.add_option('--gender',
              action='store_true', dest=gb,
              help='Whether to perform gender bias analysis. Very slow.')


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


def text_standard(flesch_reading_score,
                  smog_score,
                  flesch_kincaid_score,
                  coleman_liau_score,
                  ari_score,
                  dale_chall_score,
                  linsear_write_score,
                  gunning_fog_score):
    grade = []

    # Appending Flesch Kincaid Grade
    lower = round(flesch_kincaid_score)
    upper = math.ceil(flesch_kincaid_score)
    grade.append(int(lower))
    grade.append(int(upper))

    # Appending Flesch Reading Easy
    score = flesch_reading_score
    if score < 100 and score >= 90:
        grade.append(5)
    elif score < 90 and score >= 80:
        grade.append(6)
    elif score < 80 and score >= 70:
        grade.append(7)
    elif score < 70 and score >= 60:
        grade.append(8)
        grade.append(9)
    elif score < 60 and score >= 50:
        grade.append(10)
    elif score < 50 and score >= 40:
        grade.append(11)
    elif score < 40 and score >= 30:
        grade.append(12)
    else:
        grade.append(13)

    # Appending SMOG Index
    lower = round(smog_score)
    upper = math.ceil(smog_score)
    grade.append(int(lower))
    grade.append(int(upper))

    # Appending Coleman_Liau_Index
    lower = round(coleman_liau_score)
    upper = math.ceil(coleman_liau_score)
    grade.append(int(lower))
    grade.append(int(upper))

    # Appending Automated_Readability_Index
    lower = round(ari_score)
    upper = math.ceil(ari_score)
    grade.append(int(lower))
    grade.append(int(upper))

    # Appending Dale_Chall_Readability_Score
    lower = round(dale_chall_score)
    upper = math.ceil(dale_chall_score)
    grade.append(int(lower))
    grade.append(int(upper))

    # Appending Linsear_Write_Formula
    lower = round(linsear_write_score)
    upper = math.ceil(linsear_write_score)
    grade.append(int(lower))
    grade.append(int(upper))

    # Appending Gunning Fog Index
    lower = round(gunning_fog_score)
    upper = math.ceil(gunning_fog_score)
    grade.append(int(lower))
    grade.append(int(upper))

    # Finding the Readability Consensus based upon all the above tests
    d = dict([(x, grade.count(x)) for x in grade])
    sorted_x = sorted(d.items(), key=operator.itemgetter(1))
    final_grade = str((sorted_x)[len(sorted_x)-1])
    score = final_grade.split(',')[0].strip('(')
    return str(int(score)-1) + 'th ' + 'and ' + str(int(score)) + 'th grade'


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
if len(args) != 1:
    op.error('Usage: python NewsClassification.py <file name>')
    sys.exit(1)

file_name = sys.argv[1]

# load all data
print('Reading in data.')
df = pd.read_csv(file_name, dtype=object, keep_default_na=False)

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


# Readability & Gender Bias
if opts.gb:
    print('WARNING: Gender bias specfied. This may incur a long runtime.')
    gd = GenderDetector('us')
    override = {}
    with open('../Data/gender.csv') as f:
        for line in f:
            term, gender = line.replace(' ', '').replace('\n', '').split(',')
            override[term] = gender

for publisher in target_names:
    flesch_reading_score = 0.
    smog_score           = 0.
    flesch_kincaid_score = 0.
    coleman_liau_score   = 0.
    ari_score            = 0.
    dale_chall_score     = 0.
    linsear_write_score  = 0.
    gunning_fog_score    = 0.

    pub_txts = df.TEXT[df.PUBLISHER == publisher]
    if opts.gb:
        g_counts = {'male': 0, 'female': 0, 'unknown': 0}

    for article_txt in pub_txts:
        # Readability
        flesch_reading_score += ts.flesch_reading_ease(article_txt)
        smog_score           += ts.smog_index(article_txt)
        flesch_kincaid_score += ts.flesch_kincaid_grade(article_txt)
        coleman_liau_score   += ts.coleman_liau_index(article_txt)
        ari_score            += ts.automated_readability_index(article_txt)
        dale_chall_score     += ts.dale_chall_readability_score(article_txt)
        linsear_write_score  += ts.linsear_write_formula(article_txt)
        gunning_fog_score    += ts.gunning_fog(article_txt)

        if opts.gb:
            # Gender bias
            for term in article_txt.split():
                try:
                    guess = gd.guess(term)
                except KeyError:
                    guess = 'unknown'
                guess = override.get(term.lower(), guess)
                g_counts[guess] += 1

    flesch_reading_score /= len(pub_txts)
    smog_score           /= len(pub_txts)
    flesch_kincaid_score /= len(pub_txts)
    coleman_liau_score   /= len(pub_txts)
    ari_score            /= len(pub_txts)
    dale_chall_score     /= len(pub_txts)
    linsear_write_score  /= len(pub_txts)
    gunning_fog_score    /= len(pub_txts)

    readability_consensus = text_standard(flesch_reading_score,
                                          smog_score,
                                          flesch_kincaid_score,
                                          coleman_liau_score,
                                          ari_score,
                                          dale_chall_score,
                                          linsear_write_score,
                                          gunning_fog_score)

    print(('Average Readability Scores ({})\n'  +
           '_' * 80 + '\n'                      +
           'Flesch Reading Ease:          {}\n' +
           'Flesch Kincaid Grade:         {}\n' +
           'SMOG Index:                   {}\n' +
           'Automated Readability Index:  {}\n' +
           'Dale Chall Readability Score: {}\n' +
           'Linsear Write Score:          {}\n' +
           'Gunning Fog:                  {}\n' +
           '-' * 80 + '\n'                      +
           'Readability Consensus:        {}\n').format(
           publisher, flesch_reading_score, flesch_kincaid_score, smog_score,
           ari_score, dale_chall_score, linsear_write_score,
           gunning_fog_score, readability_consensus))

    if opts.gb:
        # Normalize gender counts
        total_terms = sum(g_counts.values())
        for k in g_counts.keys():
            g_counts[k] /= total_terms * 100

        print('Gender Term Distribution:\n' +
              ''.join('{}: {}%\n'.format(k, v) for k, v in g_counts.items()))

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
