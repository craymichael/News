from lxml import html
import requests
import sys
import pandas as pd


def scrape_news(search_term, upper_limit=None):
    search_url = ('https://www.google.com/search?q={}&tbm=nws'
                  '&start={}&sa=N&dpr=1').format(
                  search_term.replace(' ',  '+'), '{}')
    page_n = 0
    urls = []
    publishers = []
    while not upper_limit or len(urls) < upper_limit:
        print('Beginning to scrape page {}.'.format(int(page_n / 10 + 1)))
        response = requests.get(search_url.format(page_n))
        page_n += 10
        text = response.text.replace('<b>', '').replace('</b>', '')
        if response.status_code == 200:
            pagehtml = html.fromstring(text)
            a_els = pagehtml.xpath('//h3[@class="r"] \
                                    /a')
            if not len(a_els):
                break

            span_els = pagehtml.xpath('//div[@class="slp"] \
                                       /span/text()')
            urls       += [a.get('href').replace('/url?q=', '', 1).rsplit('&sa=', 1)[0]
                           for a in a_els]
            publishers += [s.rsplit(' -', 1)[0] for s in span_els]
        else:
            raise ValueError('Uh oh, scary status code: {}'.format(
                             response.status_code))

    print('Scraped a total of {} article links.'.format(len(urls)))

    return urls, publishers

if __name__ == '__main__':
    if len(sys.argv) not in {2, 3}:
        print('Usage: python ScrapeGNews.py <search phrase> [upper limit]')
        sys.exit(-1)

    search_phrase = sys.argv[1]
    if len(sys.argv) > 2:
        upper_limit = int(sys.argv[2])
    else:
        upper_limit = None

    urls, publishers = scrape_news(search_phrase, upper_limit)

    df = pd.DataFrame({'PUBLISHER': publishers,
                       'URL': urls})
    df.to_csv('{}.csv'.format(search_phrase.replace(' ', '_')), index=False)

    print('Unique Publishers:')
    print(df.PUBLISHER.value_counts())
