

from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from bs4 import BeautifulSoup
import requests

model_name = "human-centered-summarization/financial-summarization-pegasus"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

intrests = ['covid','sports','finance']
def search_news(intrests):
    search_url = "https://www.google.com/search?q=yahoo+India+{}&tbm=nws".format(intrests)
    r = requests.get(search_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    atags = soup.find_all('a')
    hrefs = [link['href'] for link in atags]
    return hrefs

raw_urls = {intrest:search_news(intrest) for intrest in intrests}

import re
exclude_list = ['maps', 'policies', 'preferences', 'accounts', 'support']
def strip_unwanted_urls(urls, exclude_list):
    val = []
    for url in urls: 
        if 'https://' in url and not any(exclude_word in url for exclude_word in exclude_list):
            res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
            val.append(res)
    return list(set(val))

cleaned_urls = {intrest:strip_unwanted_urls(raw_urls[intrest], exclude_list) for intrest in intrests}

def scrape_and_process(URLs):
    ARTICLES = []
    for url in URLs: 
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = [paragraph.text for paragraph in paragraphs]
        words = ' '.join(text).split(' ')[:350]
        ARTICLE = ' '.join(words)
        ARTICLES.append(ARTICLE)
    return ARTICLES

articles = {intrest:scrape_and_process(cleaned_urls[intrest]) for intrest in intrests}

def summarize(articles):
    summaries = []
    for article in articles:
        input_ids = tokenizer.encode(article, return_tensors='pt')
        output = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries

summaries = {intrest:summarize(articles[intrest]) for intrest in intrests}

from transformers import pipeline
sentiment = pipeline('sentiment-analysis')

scores = {intrest:sentiment(summaries[intrest]) for intrest in intrests}

def create_output_array(summaries, scores, urls):
    output = []
    for intrest in intrests:
        for counter in range(len(summaries[intrest])):
            output_this = [
                intrest,
                summaries[intrest][counter],
                scores[intrest][counter]['label'],
                scores[intrest][counter]['score'],
                urls[intrest][counter]
            ]
            output.append(output_this)
    return output

final_output = create_output_array(summaries, scores, cleaned_urls)

import csv
with open('YOUR_NEWS.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerows(final_output)

