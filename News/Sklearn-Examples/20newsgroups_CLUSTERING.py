# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

print(__doc__)
op.print_help()

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
# Uncomment the following to do the analysis on all the categories
#categories = None

print("Loading 20 newsgroups dataset for categories:")
print(categories)

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)

print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
print()

labels = dataset.target
true_k = np.unique(labels).shape[0]

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    if opts.use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=opts.n_features,
                                   stop_words='english', non_negative=True,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        vectorizer = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english',
                                       non_negative=False, norm='l2',
                                       binary=False)
else:
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words='english',
                                 use_idf=opts.use_idf)
X = vectorizer.fit_transform(dataset.data)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

if opts.n_components:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(opts.n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    print("done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print()

if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))

print()

if not opts.use_hashing:
    print("Top terms per cluster:")

    if opts.n_components:
        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()

Xnew = vectorizer.fit_transform([
        "As part of 500 marches worldwide, throngs of people braved soggy, rainy weather in the nation’s capitol on Saturday to both celebrate Earth Day and participate in a historic show of support for science. I attended the flagship march, which was held in Washington D.C., but demonstrations were worldwide, on every continent — even Antarctica. \nIt was the D.C. march that had some of the most high-profile speakers, all of whom came from diverse educational, social, and professional backgrounds. Hosted by Derek Muller and musician Questlove, the series of speeches that preceded the afternoon march covered a wide-range of inspiring topics, from astronomy to medicine to environmental science. Our Warming World: The Future of Climate Change [INFOGRAPHIC]\nClick to View Full Infographic\nSpeakers ranged from the aspiring to the esteemed figures in the scientific community.\nOne of the most inspiring of the former was Taylor Richardson, 13, who raised enough money for 1,000 young girls to go see the film Hidden Figures. Richardson, an aspiring astronaut herself, saw the film at a screening at the White House in December and was so inspired that she wanted to be sure her peers back home in Florida would have the chance as well. She ended up raising enough money for several screenings and over 700 copies of the book upon which the film was based.\nAstronauts past and present joined the march, too: Leland Melvin (perhaps most well-known to Twitter for taking his official NASA portrait with his dogs) and Anousheh Ansari, an Iranian-American engineer who became the first female “space tourist” in 2006. Consequently, she was the first Iranian astronaut as well. At 91, Dr. Nancy Roman, known as “Mother Hubble”, was the oldest honoree. In addition to her work on the Hubble Telescope, Roman was also the first female executive at NASA.\nLater in the day, Dr. Jon Foley and Dr. Michael Mann spoke about perhaps one of the most pressing issues of our time: climate change and humanity’s impact on the environment. A topic that — despite hard evidence — is still being contested. Politicians and science-deniers have worked hard to discredit not just the work of scientists, but the scientists themselves.\nThis weekend, the world marched to show their support for these scientists and the work that they do.\n\There were also several special guest speakers who spoke about the value of science beyond the realm of researcg, highlighting its importance in relation to our everyday lives. After speaking about her work that ultimately connected the water in Flint, Michigan to elevated levels of lead sickening the kids in her clinic, Dr. Mona Hanna-Attisha introduced 9-year-old Mari Copeny — known throughout the country as Little Miss Flint — who spoke passionately to the crowd about how “when we reject science, kids get hurt.”\nArtist Maya Lin, best known for creating Washington’s Vietnam War Memorial when she was just 21 years old, spoke of her latest — and final — memorial, entitled “What Is Missing?” The multi-site installation uses science-based artworks to convey the immediacy and profundity of mass extinction.\nDenis Hayes, who organized the first Earth Day in 1970, was somewhat amazed to think that he was standing in front of a group assembled to fight the same battle he began over 40 years ago. “Our job is clear,” Hayes told the crowd, “Today is the first step in a longterm battle for scientific integrity, a battle for transparency, [and] a battle for survival.”\nAnd, of course, one of the most anticipated speakers was science communicator Bill Nye, who reiterated the ongoing importance of scientific inquiry, discovery, and persistence. Nye then lead the fray as it spread exponentially through the streets of D.C., marching toward the Capitol.\nI was lucky enough to be there marching myself and spoke with several scientists, all of whom came from diverse backgrounds not just in terms of their education and careers, but their life experiences. What everyone seemed to have in common, though, was their response when I asked how they felt about leaving their work for a day in order to attend the march (in chilly, damp weather, no less). Unanimously, their answer was some variation on, “I wouldn’t miss it for the world.”\nIn truth, the world needs those tens of thousands of chanting, sopping-wet scientists. In any case, most of the marchers were well-prepared for the weather. After all — science predicted it would rain."])

predictions = km.predict(Xnew)
print(predictions)
