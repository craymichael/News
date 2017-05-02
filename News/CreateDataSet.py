import pandas as pd
from time import time
from Extract import extract_txt_from_url
from multiprocessing import Value, Array, Pool
from ctypes import c_char_p, c_double


def extract_article(url, i):
    t0_url = time()

    try:
        text = extract_txt_from_url(url)
    except Exception as e:
        print('[Row {}] Could not access page at: {}'.format(i, url))
        print('\tException Message: {}'.format(e))
        ...
        n_bad_urls.value += 1
        t_bad.value += time() - t0_url
    else:
        articles[i] = text
        n_good_urls.value += 1
        t_good.value += time() - t0_url


# Create dfs (as iterable)
dfs = pd.read_csv('uci-news-aggregator.csv',
                  usecols=('TITLE', 'URL', 'PUBLISHER'),
                  dtype=object,
                  chunksize=200)

n_bad_urls  = Value(c_int, 0)       # Number of bad URLs (e.g. due to 404)
n_good_urls = Value(c_int, 0)       # Number of URLs with good data
t0          = time()                # Start time
t_good      = Value(c_double, 0.0)  # Total time spent handling good URLs
t_bad       = Value(c_double, 0.0)  # Total time spent handling bad URLs

pool = Pool(processes=32)

# Iterate over chunks
for chunk_n, df in enumerate(dfs):
    print('Beginning work on chunk {} of data.'.format(chunk_n))

    n_rows   = df.shape[0]
    articles = Array(c_char_p, n_rows)  # MP
    results  = []

    # Extract text from URLs
    for i, url in enumerate(df.URL):  # df.URL is a copy
        if (i + 1) % 50 == 0 or i == 0:
            print('Retrieving text {} of {} ({}%).'.format(
                i + 1, n_rows, float(i + 1) / n_rows))

        # Spawn process
        results.append(
                pool.apply_async(extract_article, (url, i))
        )

    # Discover df rows to drop
    to_drop = [index.get() for index in results]

    # Append articles column to df
    df = df.assign(TEXT=articles)

    # Drop df "bad" rows
    if to_drop:
        df = df.drop(df.index[to_drop])

    t_elapsed = time() - t0

    print('-' * 80)
    print('Stats:'
          '\tTotal good URLs: {}'
          '\tTotal bad URLs:  {}'
          '\tAvg time per URL (good): {}'
          '\tAvg time per URL (bad):  {}'
          '\tElapsed time: {} s'
          '\n'.format(n_good_urls.value, n_bad_urls.value,
                      t_good.value / float(t_elapsed),
                      t_bad.value / float(t_elapsed),
                      t_elapsed))
    print('\tTotal number of "bad" URLs:  {} ({}%)'.format(n_bad_urls.value,
        n_bad_urls.value / float(n_bad_urls.value + n_good_urls.value)))
    print('\tTotal number of "good" URLs: {} ({}%)'.format(n_good_urls.value,
        n_good_urls.value / float(n_bad_urls.value + n_good_urls.value)))


    print('Appending {} rows to file.'.format(n_rows))

    # Save to file
    df.to_csv('uci-news-complete.csv', mode='a', index=False)

