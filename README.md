# Sentiment Analysis App

# Sentiment Analysis App

This application uses a state-of-the-art **RoBERTa** model (specifically trained on Tweets) to classify text into **Positive**, **Negative**, or **Neutral**.

## Technical Methods
- **Architecture**: Transformer (RoBERTa).
- **Technique**: Deep Learning & Transfer Learning.
- **Preprocessing**: Regular Expressions (Regex) for text cleaning.

## Tools & Frameworks
- **Python**: Core programming language.
- **Hugging Face Transformers**: For loading and using the pre-trained RoBERTa model.
- **PyTorch**: The deep learning framework powering the model.
- **Gradio**: For building the interactive web interface.
- **SciPy**: For mathematical operations.

## Limitations
- **English Only**: The model is trained on English tweets and will not work well for other languages.
- **Sarcasm**: Like most AI, it may struggle to detect subtle sarcasm or irony.
- **Length Limit**: It can only process up to ~512 words/tokens at a time. Longer text will be cut off.
- **Domain Specific**: It is optimized for social media text (informal, slang) and might be less accurate on formal documents.

## Model Outputs
For every piece of text you enter, the model provides:
1.  **Sentiment Label**: The classification result (**Positive**, **Negative**, or **Neutral**).
2.  **Confidence Score**: A probability score (e.g., 0.9854) indicating how certain the model is.
3.  **Cleaned Text**: The text after removing noise (URLs, emojis, etc.).

## Setup
1.  **Open a terminal** in this folder.
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Running the App
1.  **Run the script**:
    ```bash
    python src/app.py
    ```
2.  **Open your browser**:
    - The terminal will show a local URL (usually `http://127.0.0.1:7860`).
    - Click it or copy-paste it into your browser.

## Features
- **Sentiment Analysis**: Classifies text as POSITIVE or NEGATIVE.
- **Text Cleaning**: Automatically removes URLs, @mentions, and emojis before analysis.
# sentiment-analysis-app
