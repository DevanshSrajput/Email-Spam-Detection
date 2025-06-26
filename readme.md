# Email Spam Detection System

A machine learning-based SMS spam detection system that classifies text messages as spam or ham (legitimate messages) using Natural Language Processing techniques and multiple classification algorithms.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [API Deployment](#api-deployment)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a comprehensive spam detection system for SMS messages using machine learning algorithms. The system preprocesses text data, extracts features using TF-IDF vectorization, and employs multiple classification models to achieve optimal performance in distinguishing between spam and legitimate messages.

### Key Objectives
- Develop an accurate spam classification model
- Compare performance of different machine learning algorithms
- Implement a scalable text preprocessing pipeline
- Deploy the model as a REST API for real-time predictions

## 📊 Dataset

The project uses the **SMS Spam Collection Dataset** ([SMS_Spam.csv](SMS_Spam.csv)) which contains:

- **Total Messages**: 5,574 SMS messages
- **Spam Messages**: 747 (13.4%)
- **Ham Messages**: 4,827 (86.6%)
- **Format**: CSV with columns for message text and labels

### Dataset Structure
```
SMS_Spam.csv
├── Column 1: Label (spam/ham)
├── Column 2: Message text
└── Additional columns (unused)
```

## ✨ Features

### Text Preprocessing
- **Case normalization**: Convert text to lowercase
- **HTML tag removal**: Clean HTML markup from messages
- **URL removal**: Remove web links and URLs
- **Special character filtering**: Remove non-alphabetic characters
- **Stop word removal**: Filter common English stop words
- **Text stemming**: Reduce words to their root forms using Porter Stemmer

### Machine Learning Models
- **Naive Bayes (MultinomialNB)**: Primary classifier optimized for text data
- **Support Vector Machine (Linear SVM)**: Alternative classifier with calibration
- **Model comparison**: Comprehensive performance evaluation

### Feature Engineering
- **TF-IDF Vectorization**: Convert text to numerical features
- **N-gram analysis**: Support for unigrams and bigrams
- **Feature selection**: Optimize feature space for better performance

## 📁 Project Structure

```
spam_detection/
├── src/
│   ├── main.py                 # Main execution script
│   ├── data_loader.py          # Data loading and preprocessing utilities
│   ├── preprocessor.py         # Text preprocessing pipeline
│   ├── models.py              # Machine learning model implementations
│   ├── compare_models.py      # Model comparison and evaluation
│   └── deploy.py              # API deployment script
├── SMS_Spam.csv               # Main dataset
├── spam_ham_dataset.csv       # Sample dataset for testing
├── spam_classifier_model.pkl  # Trained model file
├── create_sample_dataset.py   # Sample data generation script
├── additional.py              # Additional utilities
├── requirements.txt           # Project dependencies
├── model_comparison.png       # Performance comparison visualization
├── training_time_comparison.png # Training time analysis
└── README.md                  # Project documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd spam_detection
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv spam_env
   source spam_env/bin/activate  # On Windows: spam_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Required Packages
```
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
nltk>=3.6
flask>=2.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 💻 Usage

### Training the Model

1. **Run the main script**
   ```bash
   cd src
   python main.py
   ```

2. **Model comparison**
   ```bash
   python compare_models.py
   ```

### Making Predictions

```python
from src.models import SpamClassifier
from src.preprocessor import EmailPreprocessor

# Load trained model
classifier = SpamClassifier()
classifier.load_model('spam_classifier_model.pkl')

# Preprocess and predict
preprocessor = EmailPreprocessor()
message = "Congratulations! You've won $1000!"
processed_message = preprocessor.preprocess_text(message)
prediction = classifier.predict([processed_message])

print(f"Message: {message}")
print(f"Prediction: {prediction[0]}")
```

### API Deployment

```bash
cd src
python deploy.py
```

The API will be available at `http://localhost:5000` with the following endpoint:
- **POST** `/predict`: Submit text for spam classification

## 🏗️ Model Architecture

### Data Processing Pipeline

1. **Data Loading**: Load SMS dataset using [`DataLoader`](src/data_loader.py)
2. **Text Preprocessing**: Clean and normalize text using [`EmailPreprocessor`](src/preprocessor.py)
3. **Feature Extraction**: Convert text to TF-IDF vectors
4. **Model Training**: Train multiple classifiers
5. **Evaluation**: Compare model performance metrics

### Text Preprocessing Steps

```python
def preprocess_text(self, text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize and stem
    tokens = [self.stemmer.stem(word) for word in text.split() 
              if word not in self.stop_words]
    
    return ' '.join(tokens)
```

## 📈 Performance Metrics

### Model Comparison Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 96.8% | 95.2% | 92.1% | 93.6% |
| Linear SVM | 98.1% | 97.3% | 94.8% | 96.0% |

### Key Performance Indicators
- **High Accuracy**: Both models achieve >96% accuracy
- **Low False Positives**: Minimized legitimate messages marked as spam
- **Balanced Performance**: Good precision-recall balance
- **Fast Training**: Efficient model training times

## 🌐 API Deployment

The system includes a Flask-based REST API for real-time spam detection:

### API Endpoints

**POST /predict**
```json
Request:
{
    "text": "Your message text here"
}

Response:
{
    "prediction": "spam" | "ham",
    "confidence": 0.95,
    "status": "success"
}
```

### Example Usage
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You won $1000!"}'
```

## 🔧 Configuration

### Model Parameters
- **TF-IDF Max Features**: 5000
- **N-gram Range**: (1,2) for unigrams and bigrams
- **Test Split**: 20% of dataset for validation
- **Random State**: 42 for reproducibility

### Preprocessing Settings
- **Stop Words**: English NLTK corpus
- **Stemmer**: Porter Stemmer
- **Text Encoding**: UTF-8

## 📊 Visualization

The project generates performance visualizations:
- [`model_comparison.png`](model_comparison.png): Model accuracy comparison
- [`training_time_comparison.png`](training_time_comparison.png): Training time analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 Future Enhancements

- [ ] Deep learning models (LSTM, BERT)
- [ ] Multi-language support
- [ ] Real-time model retraining
- [ ] Enhanced feature engineering
- [ ] Mobile app integration
- [ ] Batch processing capabilities

## 🐛 Troubleshooting

### Common Issues

1. **NLTK Data Missing**
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

2. **Memory Issues with Large Datasets**
   - Reduce TF-IDF max_features
   - Use batch processing for predictions

3. **Model Loading Errors**
   - Ensure model file exists in correct directory
   - Check pickle compatibility between Python versions

## 📞 Support

For questions or issues:
- Create an issue in the repository
- Contact: dksdevansh@gmail.com
- Documentation: Check inline code comments

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- SMS Spam Collection Dataset contributors
- Scikit-learn development team
- NLTK project contributors
- Open source community

---

**Author**: Devansh Singh  
**Project Type**: Machine Learning Internship Project  
**Last Updated**: 2025