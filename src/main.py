from data_loader import DataLoader
from preprocessor import EmailPreprocessor
from models import SpamClassifier
from compare_models import compare_models

def main():
    """
    Main function to run the spam detection project
    """
    # Set the path to your dataset
    # For example, using the SpamAssassin public corpus or Enron dataset
    data_path = r'C:\Users\Devansh Singh\OneDrive\Desktop\Tamizhan Skills\P1--Email_Spam\SMS_Spam.csv'  # Replace with your dataset path
    
    # Compare models
    nb_classifier, svm_classifier, metrics_df = compare_models(
        data_path, 
        text_column='text',  # Replace with your text column name
        label_column='label'  # Replace with your label column name
    )
    
    # Save the best model (assuming Naive Bayes for lightweight deployment)
    nb_classifier.save_model('spam_classifier_model.pkl')
    
    print("\nSpam detection project completed successfully!")
    print("The trained model is saved as 'spam_classifier_model.pkl'")

if __name__ == "__main__":
    main()