import pandas as pd, numpy as np
import nltk
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from bertopic.vectorizers import ClassTfidfTransformer
# from gensim.models.coherencemodel import CoherenceModel

from sentence_transformers import SentenceTransformer
from bertopic.representation import OpenAI
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from topic_modelling.CustomRepresentationModelV2 import CustomRepresentationModelV2
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence

import os
import re
from IPython.display import Markdown

nltk.download('punkt')
nltk.download('stopwords')

class BERTopicModel():
    
    def __init__(self, min_cluster_size=None, min_sample_size=None, embeddings=None, nr_topics=None, reduce_outliers=False, messages=None):
        
        """
        Initialize the BERTopicModel class.
        
        Parameters:
        - min_cluster_size (int, optional): The minimum size of clusters to consider. Default is None.
        - min_sample_size (int, optional): The minimum number of samples in a cluster. Default is None.
        - embeddings (str, optional): The name of the embedding model to use. Default is None.
        - nr_topics (int, optional): The number of topics to identify. Default is None.
        - reduce_outliers (bool, optional): Whether to reduce outliers in topics. Default is False.
        - messages (list, optional): A list of message objects for GPT API call.
        
        """
        super().__init__()
        self.hyperparameters = dict()
        self.hyperparameters['min_cluster_size'] = 150
        self.hyperparameters['min_sample_size'] = 50
        self.hyperparameters['embeddings'] = embeddings
        self.BERTopic_model = None
        self.BERTopic_topics = 40
        self.reduce_outliers = reduce_outliers
        self.messages = messages
        
        # Initialize vectorizer, UMAP, and sentence model
        vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        
        # whaleloops/phrase-bert
        # all-mpnet-base-v2
        sentence_model = SentenceTransformer('all-mpnet-base-v2', device="cuda")
        
        # Initialize parameters for BERTopic
        self.init_params = {'vectorizer_model': vectorizer_model, 'umap_model': umap_model, 'embedding_model': sentence_model}
        
        # Set min_cluster_size and min_sample_size if provided
        if self.hyperparameters['min_cluster_size'] is not None:
            hdbscan_model = HDBSCAN(metric='euclidean', cluster_selection_method='eom', prediction_data=False, 
                                    min_cluster_size=self.hyperparameters['min_cluster_size'],
                                    min_samples=self.hyperparameters['min_sample_size'])
            self.init_params['hdbscan_model'] = hdbscan_model

        # Set nr_topics if provided    
        if nr_topics is not None :
            self.init_params['nr_topics'] = nr_topics
            
        self.BERTopic_model = BERTopic(**self.init_params)
    
    def train_model(self, dataset: list):
        
        """
        Train the BERTopic model.

        Parameters:
        - dataset (list of strings): The dataset to train the BERTopic model.

        Returns:
        - result (dict): A dictionary containing topics identified by the BERTopic model.
        - freq (DataFrame): A DataFrame containing the frequency of words in topics.
        - topic_dict (dict): A dictionary containing topics and associated words.
        - BERTopic_topics (list): A list containing the identified topics.
        - BERTopic_model (object): The trained BERTopic model.
        
        """

        if self.hyperparameters['embeddings'] is not None:
            self.BERTopic_topics, _ = self.BERTopic_model.fit_transform(dataset, embeddings=self.hyperparameters['embeddings'])
        else:
            self.BERTopic_topics, _ = self.BERTopic_model.fit_transform(dataset)
        
        # Reduce outliers if enabled
        if self.reduce_outliers:
            try:
                # self.BERTopic_topics, _ = self.BERTopic_model.transform(dataset)
                self.BERTopic_topics = self.BERTopic_model.reduce_outliers(dataset, self.BERTopic_topics, strategy="c-tf-idf")
            except Exception as e:
                print('Exception while reducing topic outliers')
                # print(e)
        
        bertopic_topics = [[topicwords[0] for topicwords in self.BERTopic_model.get_topic(i)[:10]] for i in range(len(set(self.BERTopic_topics)) - 1)]
        result = dict()
        result['topics'] = bertopic_topics
        
        # Update topics if messages are provided
        if self.messages is not None:            
            representation_model = CustomRepresentationModelV2(nr_docs=30, verbose=False, messages=self.messages)
            self.BERTopic_model.update_topics(dataset, topics=self.BERTopic_topics, vectorizer_model=self.init_params['vectorizer_model'], representation_model=representation_model)
        
        freq = self.BERTopic_model.get_topic_info()
        topic_dict = self.BERTopic_model.get_topics()
        
        return result, freq, topic_dict, self.BERTopic_topics, self.BERTopic_model
    
    def evaluate_model(self, docs: list, results):
        dataset = [d.split() for d in docs]
        npmi = Coherence(texts=dataset, topk=10, measure='c_v').score(results)
        td = TopicDiversity().score(results)

        return npmi, td