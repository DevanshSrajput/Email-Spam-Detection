import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class EmailPreprocessor:
    """
    Class for preprocessing email text data
    """
    
    def __init__(self, max_features=5000, min_df=2, max_df=0.95):
        """
        Initialize the preprocessor
        
        Parameters:
        -----------
        max_features : int
            Maximum number of features (words) to keep
        min_df : int or float
            Minimum document frequency for a word to be included
        max_df : float
            Maximum document frequency for a word to be included
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words='english',
            preprocessor=self.preprocess_text
        )
    
    def preprocess_text(self, text):
        """
        Preprocess a single text
        
        Parameters:
        -----------
        text : str
            Raw email text
        
        Returns:
        --------
        str : Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove email headers
        text = re.sub(r'^.*?Subject:', '', text, flags=re.DOTALL)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize, remove stopwords and stem
        tokens = text.split()
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
        
        # Join tokens back to string
        return ' '.join(tokens)
    
    def fit_transform(self, X_train):
        """
        Fit and transform the training data
        
        Parameters:
        -----------
        X_train : array-like
            Training text data
        
        Returns:
        --------
        sparse matrix : TF-IDF features for training data
        """
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        print(f"Created {X_train_tfidf.shape[1]} features using TF-IDF vectorization")
        return X_train_tfidf
    
    def transform(self, X_test):
        """
        Transform test data
        
        Parameters:
        -----------
        X_test : array-like
            Test text data
        
        Returns:
        --------
        sparse matrix : TF-IDF features for test data
        """
        return self.vectorizer.transform(X_test)
    
    def get_feature_names(self):
        """
        Get feature names from the vectorizer
        
        Returns:
        --------
        list : Feature names
        """
        return self.vectorizer.get_feature_names_out()