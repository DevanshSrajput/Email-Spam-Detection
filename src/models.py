import numpy as np
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time

class SpamClassifier:
    """
    Spam classification model class
    """
    
    def __init__(self, classifier_type='naive_bayes', alpha=1.0, C=1.0):
        """
        Initialize the classifier
        
        Parameters:
        -----------
        classifier_type : str
            Type of classifier ('naive_bayes' or 'svm')
        alpha : float
            Smoothing parameter for Naive Bayes
        C : float
            Regularization parameter for SVM
        """
        self.classifier_type = classifier_type
        
        # Initialize the model based on classifier_type
        if classifier_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=alpha)
        elif classifier_type == 'svm':
            # Use CalibratedClassifierCV to get probability estimates from SVM
            svm = LinearSVC(C=C, max_iter=10000)
            self.model = CalibratedClassifierCV(svm)
        else:
            raise ValueError("classifier_type must be 'naive_bayes' or 'svm'")
    
    def train(self, X_train, y_train):
        """
        Train the model
        
        Parameters:
        -----------
        X_train : sparse matrix
            Training features
        y_train : array-like
            Training labels
        
        Returns:
        --------
        float : Training time in seconds
        """
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        print(f"Model training completed in {training_time:.2f} seconds")
        return training_time
    
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
        -----------
        X : sparse matrix
            Features to predict
        
        Returns:
        --------
        array : Predicted labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get probability estimates
        
        Parameters:
        -----------
        X : sparse matrix
            Features to predict
        
        Returns:
        --------
        array : Probability estimates
        """
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model
        
        Parameters:
        -----------
        X_test : sparse matrix
            Test features
        y_test : array-like
            True labels
        
        Returns:
        --------
        dict : Performance metrics
        """
        # Measure prediction time
        start_time = time.time()
        y_pred = self.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Print results
        print(f"Classification report for {self.classifier_type}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Prediction time: {prediction_time:.4f} seconds for {len(y_test)} samples")
        print(f"Confusion matrix:\n{cm}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'prediction_time': prediction_time
        }
        
    def save_model(self, file_path):
        """
        Save the model to disk
        
        Parameters:
        -----------
        file_path : str
            Path to save the model
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {file_path}")
    
    @staticmethod
    def load_model(file_path):
        """
        Load a model from disk
        
        Parameters:
        -----------
        file_path : str
            Path to load the model from
        
        Returns:
        --------
        SpamClassifier : Loaded model
        """
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {file_path}")
        return model