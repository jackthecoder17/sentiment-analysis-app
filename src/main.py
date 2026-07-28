import re
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import pipeline

# Define the request and response models
class AnalyzeRequest(BaseModel):
    texts: list[str]

class SentimentResult(BaseModel):
    label: str
    score: float
    cleaned_text: str

class AnalyzeResponse(BaseModel):
    results: list[SentimentResult | str]

app = FastAPI(title="Sentiment Analysis API")

# Mount the static directory to serve HTML, CSS, JS
import os
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

print("Loading model... please wait.")
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

def preprocess_text(text: str) -> str:
    """
    Cleans the text by removing URLs, user mentions (@user),
    and non-alphanumeric characters (like emojis), keeping basic punctuation.
    """
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\']', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_sentiment(request: AnalyzeRequest):
    """
    Analyzes the sentiment of a batch of input texts.
    Returns a list of structured results.
    """
    texts = request.texts
    cleaned_texts = [preprocess_text(t) for t in texts]
    
    # Identify strings that actually have text after cleaning
    valid_indices = [i for i, t in enumerate(cleaned_texts) if t.strip()]
    valid_cleaned = [cleaned_texts[i] for i in valid_indices]
    
    # Initialize results list
    raw_results = [None] * len(texts)
    
    if valid_cleaned:
        # Process the valid texts in the model
        batch_results = sentiment_pipeline(valid_cleaned)
        for idx, res in zip(valid_indices, batch_results):
            raw_results[idx] = res
            
    final_outputs = []
    
    for i, text in enumerate(texts):
        if not text.strip():
            final_outputs.append("Please enter some text.")
            continue
            
        cleaned = cleaned_texts[i]
        if not cleaned:
            final_outputs.append("Input contained only non-text characters (emojis/links) which were removed.")
            continue
            
        res = raw_results[i]
        label = res['label']
        score = res['score']
        
        # Humanize labels
        if label == 'LABEL_0': label = 'Negative'
        elif label == 'LABEL_1': label = 'Neutral'
        elif label == 'LABEL_2': label = 'Positive'
        else:
            label = label.capitalize()
            
        final_outputs.append(SentimentResult(
            label=label,
            score=float(score),
            cleaned_text=cleaned
        ))
        
    return AnalyzeResponse(results=final_outputs)

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(static_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
