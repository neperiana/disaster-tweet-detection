
import pandas as pd

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


