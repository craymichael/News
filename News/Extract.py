""" ==========================================================================
    Copyright 2017 Zach Carmichael

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    ==========================================================================
"""
from newspaper import Article
import time


def extract_txt_from_url(url):
    """ Extracts article text from a URL.
    Args:
        url: The URL to retrieve text from
    Returns:
        Extracted text
    """
    a = Article(url)

    a.download()
    a.parse()

    return a.title + ' ' + a.text

