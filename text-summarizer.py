#!/usr/bin/env python
# coding: utf-8
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

# Parse articles to sentences
def urlToText(the_url):

    # Download the data
    htmlData = requests.get(the_url).text
    soup = BeautifulSoup(htmlData,'html.parser')

    #title = soup.title
    #article_headline = soup.find('h1',{'class':'story-body__h1'})

    # Get the Div containing the article
    # BBC
    story_inner = soup.find('div',{'class':'story-body__inner'})
    # Forbes
    if(len(str(story_inner))<10):
        story_inner = soup.find('div',{'class':'article-body'})
    # RTE
    if(len(str(story_inner))<10):
        story_inner = soup.find('section',{'class':'article-body'})
    # HBR
    if(len(str(story_inner))<10):
        story_inner = soup.find('div',{'class':'article'})
    # Sky News
    if(len(str(story_inner))<10):
        story_inner = soup.find('div',{'class':'sdc-article-body'})
    
    # If we got a result, re-parse it
    if(len(str(story_inner))>10):
        soup = BeautifulSoup(str(story_inner), 'html.parser')

    # Find all the "<p>" elements
    article = soup.findAll('p',{'class': None})
    
    if len(article) == 0:
        article = soup.findAll('p')
        
    # Convert results to an array of innerText items
    articleText = [a.text for a in article]

    return articleText

def read_article(the_url):

    paragraphs = urlToText(the_url)
    sentences = []
    
    for paragraph in paragraphs:
        article = paragraph.split(". ")
        for sentence in article:
            if(len(sentence.strip())>0):
                sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(the_url, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences =  read_article(the_url)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize texr
    print("Summarize Text: \n", ". ".join(summarize_text))

# let's begin
#the_url = "https://www.bbc.com/news/world-europe-51815911"
#the_url = "https://www.rte.ie/news/2020/0311/1121526-harris-covid-19/"
the_url = "https://www.newsroom.co.nz/ideasroom/2020/03/12/1077762/eight-job-myths-about-artificial-intelligence"

generate_summary(the_url, 5)
