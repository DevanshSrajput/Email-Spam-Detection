import time
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import DataLoader
from preprocessor import EmailPreprocessor
from models import SpamClassifier

def compare_models(data_path, text_column='text', label_column='label'):
    """
    Compare Naive Bayes and SVM models
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset
    text_column : str
        Name of column containing email text
    label_column : str
        Name of column containing labels
    
    Returns:
    --------
    tuple : (nb_classifier, svm_classifier, metrics_comparison)
    """
    # Load and prepare data
    loader = DataLoader(data_path)
    data = loader.load_data()
    X_train, X_test, y_train, y_test = loader.prepare_data(data, text_column, label_column)
    
    # Preprocess data
    preprocessor = EmailPreprocessor()
    X_train_tfidf = preprocessor.fit_transform(X_train)
    X_test_tfidf = preprocessor.transform(X_test)
    
    # Initialize and train Naive Bayes
    print("\nTraining Naive Bayes model...")
    nb_classifier = SpamClassifier(classifier_type='naive_bayes')
    nb_time = nb_classifier.train(X_train_tfidf, y_train)
    
    # Initialize and train SVM
    print("\nTraining SVM model...")
    svm_classifier = SpamClassifier(classifier_type='svm')
    svm_time = svm_classifier.train(X_train_tfidf, y_train)
    
    # Evaluate models
    print("\nEvaluating Naive Bayes model...")
    nb_metrics = nb_classifier.evaluate(X_test_tfidf, y_test)
    
    print("\nEvaluating SVM model...")
    svm_metrics = svm_classifier.evaluate(X_test_tfidf, y_test)
    
    # Create comparison
    comparison = {
        'Model': ['Naive Bayes', 'SVM'],
        'Accuracy': [nb_metrics['accuracy'], svm_metrics['accuracy']],
        'Precision': [nb_metrics['precision'], svm_metrics['precision']],
        'Recall': [nb_metrics['recall'], svm_metrics['recall']],
        'F1 Score': [nb_metrics['f1'], svm_metrics['f1']],
        'Training Time (s)': [nb_time, svm_time],
        'Prediction Time (s)': [nb_metrics['prediction_time'], svm_metrics['prediction_time']]
    }
    
    metrics_df = pd.DataFrame(comparison)
    print("\nModel Comparison:")
    print(metrics_df)
    
    # Plot metrics comparison
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    plt.figure(figsize=(10, 6))
    metrics_df.set_index('Model')[metrics_to_plot].plot(kind='bar', ylim=(0.8, 1.0))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    
    # Plot training time comparison
    plt.figure(figsize=(8, 5))
    metrics_df.set_index('Model')['Training Time (s)'].plot(kind='bar')
    plt.title('Training Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.tight_layout()
    plt.savefig('training_time_comparison.png')
    
    return nb_classifier, svm_classifier, metrics_df