import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    """
    Class for loading and preparing the email dataset
    """
    
    def __init__(self, data_path, test_size=0.2, random_state=42):
        """
        Initialize the DataLoader
        
        Parameters:
        -----------
        data_path : str
            Path to the dataset file
        test_size : float
            Proportion of the dataset to include in the test split
        random_state : int
            Random state for reproducibility
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        
    def load_data(self):
        """
        Load data from file
        
        Returns:
        --------
        DataFrame : Pandas DataFrame containing the loaded data
        """
        # If the data is in CSV format, use:
        data = pd.read_csv(self.data_path)
        
        # For different formats, you might need different methods:
        # If TSV format: data = pd.read_csv(self.data_path, sep='\t')
        # If Excel format: data = pd.read_excel(self.data_path)
        
        print(f"Dataset loaded with {len(data)} samples")
        return data
    
    def prepare_data(self, data, text_column='text', label_column='label'):
        """
        Prepare data for training and testing
        
        Parameters:
        -----------
        data : DataFrame
            The loaded dataset
        text_column : str
            Name of column containing email text
        label_column : str
            Name of column containing labels (spam/ham)
        
        Returns:
        --------
        tuple : (X_train, X_test, y_train, y_test)
        """
        # Extract features and labels
        X = data[text_column].values
        y = data[label_column].values
        
        # Convert labels to binary (if needed)
        if not isinstance(y[0], (int, np.integer)):
            # If labels are string ('spam', 'ham')
            y = np.array([1 if label == 'spam' else 0 for label in y])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Testing set: {len(X_test)} samples")
        print(f"Spam ratio in training: {sum(y_train)/len(y_train):.2f}")
        print(f"Spam ratio in testing: {sum(y_test)/len(y_test):.2f}")
        
        return X_train, X_test, y_train, y_test