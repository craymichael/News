import pandas as pd
import sys
from time import time
from datetime import datetime
from Extract import extract_txt_from_url
from multiprocessing import Value, Pool
from ctypes import c_double, c_int
from newspaper.article import ArticleException


# Constants
MAX_ROWS = 1000
EPSILON  = sys.float_info.epsilon


def extract_article(url_i):
    url, i = url_i
    t0_url = time()

    try:
        text = extract_txt_from_url(url)
    except ArticleException as e:
        print('[Row {}] Could not access page at: {}'.format(i, url))
        print('\tException Message: {}'.format(e))
        n_bad_urls.value += 1
        t_bad.value += time() - t0_url
        text = None
    else:
        n_good_urls.value += 1
        t_good.value += time() - t0_url

    return (i, text)


# Create dfs (as iterable)
dfs = pd.read_csv('uci-news-aggregator.csv',
                  usecols=('TITLE', 'URL', 'PUBLISHER'),
                  dtype=object,
                  chunksize=MAX_ROWS)

n_bad_urls  = Value(c_int, 0)       # Number of bad URLs (e.g. due to 404)
n_good_urls = Value(c_int, 0)       # Number of URLs with good data
t0          = time()                # Start time
t_good      = Value(c_double, 0.0)  # Total time spent handling good URLs
t_bad       = Value(c_double, 0.0)  # Total time spent handling bad URLs

# Create Process Pool
pool = Pool(processes=20)

# Iterate over chunks
for chunk_n, df in enumerate(dfs):
    print('Beginning work on chunk {} of data.'.format(chunk_n))

    n_rows = df.shape[0]

    # Extract text from URLs, df.URL returns a copy
    input_data = [(url, i) for i, url in enumerate(df.URL)]

    # Spawn processes
    results = pool.imap_unordered(extract_article, input_data)

    # Discover df rows to drop
    to_drop   = []
    to_assign = [None] * n_rows
    for i, result in enumerate(results):
        index, text = result
        if (i + 1) % 50 == 0 or i == 0:
            print('-' * 80)
            print('{} result(s) processed: {}% of chunk\n'.format(
                i + 1, float(i + 1) / n_rows))

        if text is None or not len(text):
            to_drop.append(index)
        else:
            to_assign[index] = text

    # Append articles column to df
    df = df.assign(TEXT=to_assign)

    # Drop df "bad" rows
    if to_drop:
        df = df.drop(df.index[to_drop])

    t_elapsed = time() - t0

    print('-' * 80)
    print('Stats ({}):'
          '\n\tAvg time per URL (good): {}'
          '\n\tAvg time per URL (bad):  {}'
          '\n\tElapsed time: {} s'
          '\n'.format(datetime.now(),
                      t_good.value / (float(n_good_urls.value) + EPSILON),
                      t_bad.value / (float(n_bad_urls.value) + EPSILON),
                      t_elapsed))
    print('\tTotal number of "bad" URLs:  {} ({}%)'.format(n_bad_urls.value,
        n_bad_urls.value / float(n_bad_urls.value + n_good_urls.value)))
    print('\tTotal number of "good" URLs: {} ({}%)'.format(n_good_urls.value,
        n_good_urls.value / float(n_bad_urls.value + n_good_urls.value)))

    print('Appending {} rows to file.'.format(n_rows))

    if not df.empty:
        # Save to file
        df.to_csv('uci-news-inchunks.csv', mode='a', index=False)

