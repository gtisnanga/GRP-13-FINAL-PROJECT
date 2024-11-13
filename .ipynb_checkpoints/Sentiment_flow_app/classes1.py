import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize the lemmatizer and define stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))