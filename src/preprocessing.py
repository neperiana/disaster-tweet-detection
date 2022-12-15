
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.corpus import stopwords


def preprocess_text(text):
    """
    Pre-processes a raw string of text: remove punctuation and numbers, 
    lemmatise and remove stop words.

    Parameters
    ----------
    text : string
        The input string of raw text.

    Returns
    -------
    clean_text : string
        Returns the pre-processed string of text.
    """
    # Extracts words for each sentence
    sentences = sent_tokenize(text)
    sentence_tokens = [word_tokenize(sentence.lower()) for sentence in sentences]

    # Flattens list of lists as a list, that includes all tokens
    tokens = [item for sublist in sentence_tokens for item in sublist]

    # Removes isolated punctuation
    tokens = [word for word in tokens if word.isalnum()]
    tokens = ["X" if word.isdigit() else word for word in tokens]

    # Initialise lemmatiser and run through tokens
    lemmatiser = WordNetLemmatizer()
    lem_tokens = [lemmatiser.lemmatize(t) for t in tokens]

    # removing stopwords from lem_tokens
    stop_words = stopwords.words('english') + ['http', 'amp']
    lem_tokens = [t for t in lem_tokens if t not in stop_words]

    return ' '.join(lem_tokens)


class TextToFeatures(object):
    """
        Convert a collection of text documents to a vector of features.

        Parameters
        ----------
        type : str, default='tfidf-vec'
            - If `'tfidf-vec'`, features are extracted using tf-idf weights up 
            to the maximum features specified.
            - If `'sentence-BERT'`, features are extracted using sentence BERT.
        max_features : int, default=250
            Max number of features to extract for tf-idf.
    """
    valid_types = ['tfidf-vec', 'count-vec', 'sentence-BERT']

    def __init__(self, type='tfidf-vec', max_features=250):
        assert type in self.valid_types, f"Type is {type} and should be one of {self.valid_types}"
        self._type = type
        self._max_features = max_features
        self._transform_ready = False

    def fit_transform(self, raw_documents):
        """
            Learn the text features and return document-feature matrix.

            Parameters
            ----------
            raw_documents : iterable
                An iterable which generates either str, unicode or file objects.
                
            Returns
            -------
            X : array of shape (n_samples, n_features)
                Document-feature matrix.
        """
        self._fit(raw_documents)
        return self._transform(raw_documents)

    def transform(self, raw_documents):
        """
            Transform documents to document-feature matrix.

            Parameters
            ----------
            raw_documents : iterable
                An iterable which generates either str, unicode or file objects.

            Returns
            -------
            X : sparse matrix of shape (n_samples, n_features)
                Document-feature matrix.
        """
        assert self._transform_ready, "TextToFeatures needs to be fit before using transform."
        return self._transform(raw_documents)


    def _fit(self, raw_documents):
        self._transform_ready = True

        if self._type == 'tfidf-vec':
            self._fit_tfidf(raw_documents)
        elif self._type == 'count-vec':
            self._fit_count(raw_documents)
        elif self._type == 'sentence-BERT':
            self._fit_sentence_BERT(raw_documents)
    
    def _transform(self, raw_documents):
        if self._type == 'tfidf-vec':
            return self._transform_tfidf(raw_documents)
        elif self._type == 'count-vec':
            return self._transform_count(raw_documents)
        elif self._type == 'sentence-BERT':
            return self._transform_sentence_BERT(raw_documents)

    def _fit_tfidf(self, raw_documents):
        self._vectorizer = TfidfVectorizer(max_features=self._max_features) 
        self._vectorizer.fit(raw_documents)
    
    def _transform_tfidf(self, raw_documents):
        return self._vectorizer.transform(raw_documents).toarray()

    def _fit_count(self, raw_documents):
        self._vectorizer = CountVectorizer(max_features=self._max_features) 
        self._vectorizer.fit(raw_documents)
    
    def _transform_count(self, raw_documents):
        return self._vectorizer.transform(raw_documents).toarray()

    def _fit_sentence_BERT(self, raw_documents):
        self._vectorizer = SentenceTransformer('all-MiniLM-L6-v2')
    
    def _transform_sentence_BERT(self, raw_documents):
        return pd.DataFrame(raw_documents).apply(
            lambda row: self._vectorizer.encode(row['clean_text']),
            axis='columns',
            result_type='expand',
        )
