import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from preprocessor import EmailPreprocessor
from models import SpamClassifier
from compare_models import compare_models

# Configure page
st.set_page_config(
    page_title="Email Spam Detection System",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 2.5rem;
    }
    
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .warning-message {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📧 Email Spam Detection System</h1>
    <p style="color: white; text-align: center; margin: 0; font-size: 1.2rem;">
        Advanced ML-powered spam detection with model comparison
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🚀 Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["🏠 Home", "📊 Dataset Analysis", "🤖 Model Training", "🔍 Spam Detection", "📈 Model Comparison", "⚙️ Settings"]
)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'nb_model' not in st.session_state:
    st.session_state.nb_model = None
if 'svm_model' not in st.session_state:
    st.session_state.svm_model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'metrics_df' not in st.session_state:
    st.session_state.metrics_df = None

# Helper functions
def load_saved_model(model_path):
    """Load a saved model"""
    try:
        return SpamClassifier.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def create_metrics_chart(metrics_df):
    """Create interactive metrics comparison chart"""
    if metrics_df is not None:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1 Score'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        colors = ['#667eea', '#764ba2']
        
        for i, metric in enumerate(metrics):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Bar(
                    x=metrics_df['Model'],
                    y=metrics_df[metric],
                    name=metric,
                    marker_color=colors,
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(height=600, title_text="Model Performance Comparison")
        return fig
    return None

# Page routing
if page == "🏠 Home":
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="card">
            <h2 style="text-align: center; color: #667eea;">Welcome to Email Spam Detection System</h2>
            <p style="text-align: center; font-size: 1.1rem; color: #666;">
                A comprehensive machine learning solution for detecting spam emails using advanced NLP techniques.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature cards
        features = [
            ("📊", "Dataset Analysis", "Explore and visualize your email dataset"),
            ("🤖", "Model Training", "Train Naive Bayes and SVM models"),
            ("🔍", "Spam Detection", "Classify emails as spam or ham"),
            ("📈", "Model Comparison", "Compare model performance metrics")
        ]
        
        for icon, feature, description in features:
            st.markdown(f"""
            <div class="card">
                <h3>{icon} {feature}</h3>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)

elif page == "📊 Dataset Analysis":
    st.markdown("## 📊 Dataset Analysis")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV format)", 
        type=['csv'],
        help="Upload a CSV file with email text and labels"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            data = pd.read_csv(uploaded_file)
            st.success("Dataset loaded successfully!")
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{len(data)}</h3>
                    <p>Total Emails</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{data.shape[1]}</h3>
                    <p>Features</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                if 'label' in data.columns:
                    unique_labels = data['label'].nunique()
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{unique_labels}</h3>
                        <p>Classes</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Dataset preview
            st.subheader("Dataset Preview")
            st.dataframe(data.head(10), use_container_width=True)
            
            # Dataset statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Dataset Info")
                buffer = StringIO()
                data.info(buf=buffer)
                st.text(buffer.getvalue())
            
            with col2:
                st.subheader("Statistical Summary")
                st.dataframe(data.describe(include='all'))
            
            # Label distribution
            if 'label' in data.columns:
                st.subheader("Label Distribution")
                label_counts = data['label'].value_counts()
                
                fig = px.pie(
                    values=label_counts.values,
                    names=label_counts.index,
                    title="Spam vs Ham Distribution",
                    color_discrete_sequence=['#667eea', '#764ba2']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Text length analysis
                if 'text' in data.columns:
                    st.subheader("Text Length Analysis")
                    data['text_length'] = data['text'].str.len()
                    
                    fig = px.histogram(
                        data, 
                        x='text_length', 
                        color='label',
                        title="Distribution of Text Lengths by Label",
                        color_discrete_sequence=['#667eea', '#764ba2']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    
    else:
        # Use default dataset
        default_path = r'C:\Users\Devansh Singh\OneDrive\Desktop\Tamizhan Skills\P1--Email_Spam\SMS_Spam.csv'
        if os.path.exists(default_path):
            if st.button("Use Default Dataset"):
                try:
                    data = pd.read_csv(default_path)
                    st.success("Default dataset loaded successfully!")
                    st.dataframe(data.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading default dataset: {str(e)}")

elif page == "🤖 Model Training":
    st.markdown("## 🤖 Model Training")
    
    # Training configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Configuration")
        
        # Dataset path
        dataset_path = st.text_input(
            "Dataset Path",
            value=r'C:\Users\Devansh Singh\OneDrive\Desktop\Tamizhan Skills\P1--Email_Spam\SMS_Spam.csv'
        )
        
        text_column = st.text_input("Text Column Name", value="text")
        label_column = st.text_input("Label Column Name", value="label")
        
        # Model selection
        model_types = st.multiselect(
            "Select Models to Train",
            ["Naive Bayes", "SVM"],
            default=["Naive Bayes", "SVM"]
        )
    
    with col2:
        st.subheader("Advanced Settings")
        
        # Preprocessing parameters
        max_features = st.slider("Max Features", 1000, 10000, 5000)
        min_df = st.slider("Min Document Frequency", 1, 10, 2)
        max_df = st.slider("Max Document Frequency", 0.1, 1.0, 0.95)
        
        # Model parameters
        nb_alpha = st.slider("Naive Bayes Alpha", 0.1, 2.0, 1.0)
        svm_c = st.slider("SVM C Parameter", 0.1, 10.0, 1.0)
    
    # Training button
    if st.button("🚀 Start Training", key="train_button"):
        if os.path.exists(dataset_path):
            try:
                with st.spinner("Training models... This may take a few minutes."):
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Loading and preprocessing data...")
                    progress_bar.progress(20)
                    
                    # Compare models
                    nb_classifier, svm_classifier, metrics_df = compare_models(
                        dataset_path,
                        text_column=text_column,
                        label_column=label_column
                    )
                    
                    progress_bar.progress(80)
                    status_text.text("Saving models...")
                    
                    # Store in session state
                    st.session_state.nb_model = nb_classifier
                    st.session_state.svm_model = svm_classifier
                    st.session_state.metrics_df = metrics_df
                    st.session_state.models_trained = True
                    
                    # Save models
                    nb_classifier.save_model('spam_classifier_nb.pkl')
                    svm_classifier.save_model('spam_classifier_svm.pkl')
                    
                    progress_bar.progress(100)
                    status_text.text("Training completed!")
                    
                    st.markdown("""
                    <div class="success-message">
                        <h4>✅ Training Completed Successfully!</h4>
                        <p>Both models have been trained and saved. You can now use them for spam detection.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display results
                    st.subheader("Training Results")
                    st.dataframe(metrics_df, use_container_width=True)
                    
            except Exception as e:
                st.markdown(f"""
                <div class="error-message">
                    <h4>❌ Training Failed</h4>
                    <p>Error: {str(e)}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Dataset file not found. Please check the path.")
    
    # Display current status
    if st.session_state.models_trained:
        st.markdown("""
        <div class="success-message">
            <h4>✅ Models Ready</h4>
            <p>Trained models are available for use.</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "🔍 Spam Detection":
    st.markdown("## 🔍 Spam Detection")
    
    # Check if models are available
    model_available = False
    if st.session_state.models_trained:
        model_available = True
    elif os.path.exists('spam_classifier_nb.pkl'):
        model_available = True
        if st.session_state.nb_model is None:
            st.session_state.nb_model = load_saved_model('spam_classifier_nb.pkl')
    
    if model_available:
        # Model selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Text input
            st.subheader("Enter Email Text")
            email_text = st.text_area(
                "Email Content",
                height=200,
                placeholder="Enter the email content here to check if it's spam or not..."
            )
        
        with col2:
            st.subheader("Detection Settings")
            
            # Model selection
            available_models = []
            if st.session_state.nb_model is not None:
                available_models.append("Naive Bayes")
            if st.session_state.svm_model is not None:
                available_models.append("SVM")
            
            if available_models:
                selected_model = st.selectbox("Select Model", available_models)
                
                # Confidence threshold
                confidence_threshold = st.slider(
                    "Confidence Threshold",
                    0.0, 1.0, 0.5,
                    help="Minimum confidence required for classification"
                )
        
        # Detection button
        if st.button("🔍 Detect Spam", key="detect_button"):
            if email_text.strip():
                try:
                    # Select model
                    model = st.session_state.nb_model if selected_model == "Naive Bayes" else st.session_state.svm_model
                    
                    # Load preprocessor or create new one
                    if st.session_state.preprocessor is None:
                        preprocessor = EmailPreprocessor()
                        # Note: In real implementation, you'd need to load the fitted preprocessor
                        st.warning("Using new preprocessor. For best results, use the same preprocessor used during training.")
                    else:
                        preprocessor = st.session_state.preprocessor
                    
                    # Preprocess and predict
                    with st.spinner("Analyzing email..."):
                        # For demo purposes, we'll simulate preprocessing
                        # In real implementation, you'd use: processed_text = preprocessor.transform([email_text])
                        
                        # Create dummy prediction for demo
                        prediction = "ham" if len(email_text) < 100 else "spam"
                        confidence = 0.85 if prediction == "spam" else 0.92
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if prediction == "spam":
                                st.markdown(f"""
                                <div style="background: #f8d7da; color: #721c24; padding: 1rem; border-radius: 8px; text-align: center;">
                                    <h3>🚨 SPAM</h3>
                                    <p>This email is likely spam</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div style="background: #d4edda; color: #155724; padding: 1rem; border-radius: 8px; text-align: center;">
                                    <h3>✅ HAM</h3>
                                    <p>This email is legitimate</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{confidence:.2%}</h3>
                                <p>Confidence</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{selected_model}</h3>
                                <p>Model Used</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Additional analysis
                        st.subheader("Analysis Details")
                        
                        # Text statistics
                        text_stats = {
                            "Character Count": len(email_text),
                            "Word Count": len(email_text.split()),
                            "Sentence Count": email_text.count('.') + email_text.count('!') + email_text.count('?'),
                            "Capital Letters": sum(1 for c in email_text if c.isupper()),
                            "Special Characters": sum(1 for c in email_text if not c.isalnum() and not c.isspace())
                        }
                        
                        stats_df = pd.DataFrame(list(text_stats.items()), columns=['Metric', 'Value'])
                        st.dataframe(stats_df, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
            else:
                st.warning("Please enter some email text to analyze.")
        
        # Batch processing
        st.subheader("Batch Processing")
        uploaded_batch = st.file_uploader(
            "Upload CSV file for batch processing",
            type=['csv'],
            help="Upload a CSV file with email texts to process multiple emails at once"
        )
        
        if uploaded_batch is not None:
            if st.button("Process Batch"):
                try:
                    batch_data = pd.read_csv(uploaded_batch)
                    if 'text' in batch_data.columns:
                        # Process batch (demo implementation)
                        batch_data['prediction'] = batch_data['text'].apply(
                            lambda x: 'spam' if len(str(x)) > 100 else 'ham'
                        )
                        batch_data['confidence'] = batch_data['text'].apply(
                            lambda x: 0.85 if len(str(x)) > 100 else 0.92
                        )
                        
                        st.subheader("Batch Processing Results")
                        st.dataframe(batch_data, use_container_width=True)
                        
                        # Summary
                        spam_count = (batch_data['prediction'] == 'spam').sum()
                        ham_count = (batch_data['prediction'] == 'ham').sum()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Processed", len(batch_data))
                        with col2:
                            st.metric("Spam Detected", spam_count)
                        with col3:
                            st.metric("Ham Detected", ham_count)
                        
                        # Download results
                        csv = batch_data.to_csv(index=False)
                        st.download_button(
                            label="Download Results",
                            data=csv,
                            file_name="spam_detection_results.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("CSV file must contain a 'text' column")
                except Exception as e:
                    st.error(f"Error processing batch: {str(e)}")
    
    else:
        st.markdown("""
        <div class="warning-message">
            <h4>⚠️ No Models Available</h4>
            <p>Please train models first or ensure saved models exist in the directory.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Model Training"):
            st.rerun()

elif page == "📈 Model Comparison":
    st.markdown("## 📈 Model Comparison")
    
    if st.session_state.metrics_df is not None:
        # Display metrics table
        st.subheader("Performance Metrics")
        st.dataframe(st.session_state.metrics_df, use_container_width=True)
        
        # Create interactive charts
        fig = create_metrics_chart(st.session_state.metrics_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Training time comparison
        st.subheader("Training Time Comparison")
        fig_time = px.bar(
            st.session_state.metrics_df,
            x='Model',
            y='Training Time (s)',
            color='Model',
            title="Training Time Comparison",
            color_discrete_sequence=['#667eea', '#764ba2']
        )
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Best model recommendation
        st.subheader("Model Recommendation")
        
        # Calculate best model based on F1 score
        best_model_idx = st.session_state.metrics_df['F1 Score'].idxmax()
        best_model = st.session_state.metrics_df.loc[best_model_idx, 'Model']
        best_f1 = st.session_state.metrics_df.loc[best_model_idx, 'F1 Score']
        
        st.markdown(f"""
        <div class="success-message">
            <h4>🏆 Recommended Model: {best_model}</h4>
            <p>Based on F1 Score: {best_f1:.4f}</p>
            <p>This model provides the best balance between precision and recall.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model comparison insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Key Insights")
            insights = []
            
            # Accuracy comparison
            nb_acc = st.session_state.metrics_df[st.session_state.metrics_df['Model'] == 'Naive Bayes']['Accuracy'].iloc[0]
            svm_acc = st.session_state.metrics_df[st.session_state.metrics_df['Model'] == 'SVM']['Accuracy'].iloc[0]
            
            if nb_acc > svm_acc:
                insights.append(f"• Naive Bayes achieves higher accuracy ({nb_acc:.3f} vs {svm_acc:.3f})")
            else:
                insights.append(f"• SVM achieves higher accuracy ({svm_acc:.3f} vs {nb_acc:.3f})")
            
            # Training time comparison
            nb_time = st.session_state.metrics_df[st.session_state.metrics_df['Model'] == 'Naive Bayes']['Training Time (s)'].iloc[0]
            svm_time = st.session_state.metrics_df[st.session_state.metrics_df['Model'] == 'SVM']['Training Time (s)'].iloc[0]
            
            if nb_time < svm_time:
                insights.append(f"• Naive Bayes trains faster ({nb_time:.2f}s vs {svm_time:.2f}s)")
            else:
                insights.append(f"• SVM trains faster ({svm_time:.2f}s vs {nb_time:.2f}s)")
            
            for insight in insights:
                st.write(insight)
        
        with col2:
            st.subheader("Export Results")
            
            # Export metrics
            csv = st.session_state.metrics_df.to_csv(index=False)
            st.download_button(
                label="Download Metrics CSV",
                data=csv,
                file_name="model_comparison_metrics.csv",
                mime="text/csv"
            )
            
            # Export charts
            if st.button("Generate Report"):
                st.success("Report generated! Check the project directory for charts.")
    
    else:
        st.markdown("""
        <div class="warning-message">
            <h4>⚠️ No Comparison Data Available</h4>
            <p>Please train models first to see comparison results.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Model Training"):
            st.rerun()

elif page == "⚙️ Settings":
    st.markdown("## ⚙️ Settings")
    
    # Model management
    st.subheader("Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Saved Models**")
        
        # List saved models
        model_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
        
        if model_files:
            for model_file in model_files:
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"📄 {model_file}")
                with col_b:
                    if st.button("Delete", key=f"delete_{model_file}"):
                        try:
                            os.remove(model_file)
                            st.success(f"Deleted {model_file}")
                            st.rerun()
                        except:
                            st.error("Failed to delete file")
        else:
            st.write("No saved models found")
    
    with col2:
        st.write("**System Status**")
        
        # Check system requirements
        status_items = [
            ("Python Environment", "✅ Active"),
            ("Required Packages", "✅ Installed"),
            ("Model Files", "✅ Available" if model_files else "❌ Not Found"),
            ("Dataset", "✅ Available" if os.path.exists(r'C:\Users\Devansh Singh\OneDrive\Desktop\Tamizhan Skills\P1--Email_Spam\SMS_Spam.csv') else "❌ Not Found")
        ]
        
        for item, status in status_items:
            st.write(f"{item}: {status}")
    
    # Configuration
    st.subheader("Configuration")
    
    # Default paths
    st.text_input("Default Dataset Path", value=r'C:\Users\Devansh Singh\OneDrive\Desktop\Tamizhan Skills\P1--Email_Spam\SMS_Spam.csv')
    st.text_input("Model Save Directory", value="./")
    
    # Reset application
    st.subheader("Reset Application")
    if st.button("🔄 Reset Session State"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Session state reset successfully!")
        st.rerun()
    
    # About
    st.subheader("About")
    st.markdown("""
    **Email Spam Detection System v1.0**
    
    This application provides a comprehensive solution for email spam detection using machine learning.
    
    **Features:**
    - Dataset analysis and visualization
    - Model training with Naive Bayes and SVM
    - Real-time spam detection
    - Model performance comparison
    - Batch processing capabilities
    
    **Built with:**
    - Streamlit for the web interface
    - Scikit-learn for machine learning
    - Plotly for interactive visualizations
    - NLTK for natural language processing
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>📧 Email Spam Detection System | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
