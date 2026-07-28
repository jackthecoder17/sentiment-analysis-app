import os
import re
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def preprocess_text(text: str) -> str:
    """
    Cleans the text by removing URLs, user mentions (@user),
    and non-alphanumeric characters (like emojis), keeping basic punctuation.
    Matches the preprocessing in the main application.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\']', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def map_model_label(label: str) -> str:
    """
    Maps the model's raw labels (which could be LABEL_X, numeric strings, or lowercase words)
    to a standardized 'Positive', 'Negative', or 'Neutral' label.
    """
    label = str(label).strip()
    if label in ['LABEL_0', '0', '0.0', 'negative', 'Negative']:
        return 'Negative'
    elif label in ['LABEL_1', '1', '1.0', 'neutral', 'Neutral']:
        return 'Neutral'
    elif label in ['LABEL_2', '2', '2.0', 'positive', 'Positive']:
        return 'Positive'
    
    # Fallback to capitalization
    return label.capitalize()

def run_evaluation(data_df, text_column, label_column, batch_size=32):
    """
    Performs sentiment analysis on the dataset and prints/plots evaluation metrics.
    """
    print(f"Total records to evaluate: {len(data_df)}")
    
    # Check if GPU is available
    device = 0 if torch.cuda.is_available() else -1
    device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    print(f"Using device: {device_name}")
    
    # Load model
    print("Loading CardiffNLP Twitter RoBERTa Sentiment model...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=device
    )
    
    # Standardize ground-truth labels in dataset
    y_true = data_df[label_column].astype(str).apply(map_model_label).tolist()
    
    # Preprocess text column
    print("Cleaning text data...")
    cleaned_texts = data_df[text_column].astype(str).apply(preprocess_text).tolist()
    
    # Handle any empty strings after cleaning
    # Fallback to original text if cleaning emptied it completely
    for i, cleaned in enumerate(cleaned_texts):
        if not cleaned.strip():
            cleaned_texts[i] = data_df[text_column].iloc[i]
            
    print("Running predictions (this may take a moment)...")
    y_pred = []
    
    # Run predictions in batches
    for i in range(0, len(cleaned_texts), batch_size):
        batch = cleaned_texts[i:i + batch_size]
        try:
            batch_results = sentiment_pipeline(batch)
            for res in batch_results:
                y_pred.append(map_model_label(res['label']))
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {e}")
            # Fallback one-by-one for this batch if batching fails
            for text in batch:
                try:
                    res = sentiment_pipeline(text)[0]
                    y_pred.append(map_model_label(res['label']))
                except Exception:
                    y_pred.append('Neutral') # Default fallback
        
        if (i + batch_size) % (batch_size * 5) == 0 or (i + batch_size) >= len(cleaned_texts):
            progress = min(i + batch_size, len(cleaned_texts))
            print(f"Processed {progress}/{len(cleaned_texts)} records...")
            
    # Calculate and print metrics
    print("\n" + "="*50)
    print("               EVALUATION REPORT")
    print("="*50)
    
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    print(f"Unique Labels in True: {set(y_true)}")
    print(f"Unique Labels in Pred: {set(y_pred)}")
    
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("-"*50)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # Compute Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=['Negative', 'Neutral', 'Positive'])
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=['Negative', 'Neutral', 'Positive'],
        yticklabels=['Negative', 'Neutral', 'Positive']
    )
    plt.title('Sentiment Analysis Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save the plot
    output_plot = 'confusion_matrix.png'
    plt.savefig(output_plot, dpi=300)
    print(f"Confusion matrix plot saved as '{output_plot}'")
    plt.show()

def get_dummy_dataset():
    """
    Creates a small representative dataset for quick testing.
    """
    data = {
        'text': [
            "I love this product! It works perfectly and saves me so much time.",
            "Absolutely terrible experience. The app crashed twice and lost my progress.",
            "It is okay, works as expected but nothing revolutionary.",
            "The customer service was friendly, but they couldn't solve my issue.",
            "Best purchase I've made all year!",
            "I hate how complicated the setup is, took me hours.",
            "We had lunch here. The food was mediocre, and the price was high.",
            "The package arrived on time. Good job!",
            "It is raining today.",
            "Not bad, but they could improve the user interface."
        ],
        'sentiment': [
            "Positive",
            "Negative",
            "Neutral",
            "Neutral",
            "Positive",
            "Negative",
            "Negative",
            "Positive",
            "Neutral",
            "Positive"
        ]
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Twitter RoBERTa sentiment model on a dataset.")
    parser.add_argument("--file", type=str, help="Path to the CSV or Excel file containing the data. If not provided, runs on sample data.")
    parser.add_argument("--text_col", type=str, default="text", help="Name of the text column in the dataset.")
    parser.add_argument("--label_col", type=str, default="sentiment", help="Name of the ground-truth sentiment column in the dataset.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for model inference.")
    
    args = parser.parse_args()
    
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File '{args.file}' not found.")
            exit(1)
        print(f"Loading dataset from: {args.file}")
        
        # Detect file format and read
        if args.file.endswith(('.xlsx', '.xls')):
            # Ensure openpyxl is installed for modern Excel files
            try:
                df = pd.read_excel(args.file)
            except ImportError:
                print("Error: 'openpyxl' library is required to read Excel files. Run: pip install openpyxl")
                exit(1)
        else:
            df = pd.read_csv(args.file)
            
        if args.text_col not in df.columns or args.label_col not in df.columns:
            print(f"Error: Columns '{args.text_col}' or '{args.label_col}' not found in the dataset.")
            print(f"Available columns: {list(df.columns)}")
            exit(1)
    else:
        print("No file provided. Running evaluation on a built-in sample dataset...")
        df = get_dummy_dataset()
        args.text_col = 'text'
        args.label_col = 'sentiment'
        
    run_evaluation(df, args.text_col, args.label_col, batch_size=args.batch_size)
