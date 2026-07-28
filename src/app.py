import gradio as gr
from transformers import pipeline

# Load the sentiment analysis pipeline
# Load the sentiment analysis pipeline
# Using a model specifically trained on Tweets that supports Positive, Negative, AND Neutral
print("Loading model... please wait.")
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

import re

def preprocess_text(text):
    """
    Cleans the text by removing URLs, user mentions (@user),
    and non-alphanumeric characters (like emojis), keeping basic punctuation.
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user @ references and '#' from hashtags
    text = re.sub(r'\@\w+|\#', '', text)
    # Remove emojis and other special characters (keeping letters, numbers, spaces, and basic punctuation)
    # This regex keeps a-z, A-Z, 0-9, spaces, and .,!?
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\']', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_sentiment(texts):
    """
    Analyzes the sentiment of a batch of input texts.
    Returns a list of formatted result strings.
    """
    cleaned_texts = [preprocess_text(t) for t in texts]
    
    # Identify strings that actually have text after cleaning
    valid_indices = [i for i, t in enumerate(cleaned_texts) if t.strip()]
    valid_cleaned = [cleaned_texts[i] for i in valid_indices]
    
    results = [None] * len(texts)
    
    if valid_cleaned:
        # Process the valid texts in the model as a batch
        # The dynamic batch_size will be up to the max_batch_size configured in Gradio
        batch_results = sentiment_pipeline(valid_cleaned, batch_size=len(valid_cleaned))
        for idx, res in zip(valid_indices, batch_results):
            results[idx] = res
            
    output_texts = []
    
    for i, text in enumerate(texts):
        if not text.strip():
            output_texts.append("Please enter some text.")
            continue
            
        cleaned = cleaned_texts[i]
        if not cleaned:
            output_texts.append("Input contained only non-text characters (emojis/links) which were removed.")
            continue
            
        # Format explicitly parsed results
        res = results[i]
        label = res['label']
        score = res['score']
        
        if label == 'LABEL_0': label = 'Negative'
        elif label == 'LABEL_1': label = 'Neutral'
        elif label == 'LABEL_2': label = 'Positive'
        else:
            label = label.capitalize()
            
        output_texts.append(f"Original: {text}\nCleaned: {cleaned}\n\nSentiment: {label}\nConfidence: {score:.4f}")
        
    return (output_texts,)



# Create the Gradio interface using Blocks for better control and CSS support
with gr.Blocks() as iface:
    gr.Markdown("# Sentiment Analysis Demo")
    gr.Markdown("This model uses a fine-tuned RoBERTa model to classify text as POSITIVE, NEGATIVE, or NEUTRAL.")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(lines=3, placeholder="Type something here (e.g., 'I love this movie!')...", label="Input Text")
            submit_btn = gr.Button("Analyze Sentiment")
        with gr.Column():
            output_text = gr.Textbox(label="Analysis Result", elem_id="output-box", lines=10)
    
    # Examples
    gr.Examples(
        examples=[
            ["I absolutely love this product! It's amazing."],
            ["I am very disappointed with the service."],
            ["It was okay, nothing special."]
        ],
        inputs=input_text
    )
    
    submit_btn.click(fn=analyze_sentiment, inputs=input_text, outputs=output_text, batch=True, max_batch_size=8)

if __name__ == "__main__":
    iface.launch(share=True)
