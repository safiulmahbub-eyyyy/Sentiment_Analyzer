"""
Load Amazon Electronics Reviews into Supabase
Simplified version - smaller dataset, faster processing

Usage:
    python scripts/load_huggingface_dataset.py
"""

import os
import sys
import time
import subprocess
from datetime import datetime

print("="*60)
print("AMAZON REVIEWS DATA LOADER (FAST VERSION)")
print("="*60)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================
# Step 1: Load data from existing temp file or download
# ============================================================
print("\n[STEP 1] Loading dataset...")

temp_file = "/tmp/amazon_reviews.csv"

if not os.path.exists(temp_file):
    print("Downloading dataset...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pandas", "requests", "-q"], check=True)
    
    import requests
    dataset_url = "https://huggingface.co/datasets/stephaniestv/Electronics_Product_Review_With_Sentiment/resolve/main/amazon_electronics_review_sentiment.csv"
    
    response = requests.get(dataset_url, stream=True)
    response.raise_for_status()
    
    with open(temp_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8*1024*1024):
            if chunk:
                f.write(chunk)
    print(f"[OK] Downloaded to {temp_file}")

import pandas as pd
df = pd.read_csv(temp_file)
print(f"[OK] Loaded {len(df):,} reviews")

# ============================================================
# Step 2: Use smaller dataset (5000 reviews for speed)
# ============================================================
MAX_REVIEWS = 5000
print(f"\n[STEP 2] Using {MAX_REVIEWS:,} reviews (smaller for faster processing)")
df = df.head(MAX_REVIEWS)

# ============================================================
# Step 3: Generate embeddings with ONNX (faster)
# ============================================================
print("\n[STEP 3] Generating embeddings...")

# Install ONNX runtime for faster inference
subprocess.run([sys.executable, "-m", "pip", "install", "onnxruntime", "transformers", "-q"], check=True)

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Process in smaller batches for speed
BATCH_SIZE = 32
texts = [f"{str(row['title'])[:500]} {str(row['text'])[:500]}"[:512] for _, row in df.iterrows()]

print(f"Generating {len(texts)} embeddings...")
start_time = time.time()
all_embeddings = []

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i+BATCH_SIZE]
    encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
    
    with torch.no_grad():
        model_output = model(**encoded)
    
    embeddings = mean_pooling(model_output, encoded['attention_mask'])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    all_embeddings.append(embeddings.numpy())
    
    if (i // BATCH_SIZE + 1) % 20 == 0:
        print(f"  Progress: {min(i+BATCH_SIZE, len(texts))}/{len(texts)}")

embeddings = np.vstack(all_embeddings)
print(f"[OK] Generated in {time.time() - start_time:.1f}s")

# ============================================================
# Step 4: Transform and insert
# ============================================================
print("\n[STEP 4] Inserting into Supabase...")

subprocess.run([sys.executable, "-m", "pip", "install", "supabase", "-q"], check=True)

from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_SERVICE_KEY'))
print("[OK] Connected to Supabase")

# Build posts
posts = []
for idx, (_, row) in enumerate(df.iterrows()):
    asin = str(row.get('asin', 'unknown'))
    timestamp = row.get('timestamp', 0)
    try:
        ts = float(timestamp) if pd.notna(timestamp) else 0
        if ts > 1000000000000:  # milliseconds
            ts = ts / 1000
        created = datetime.fromtimestamp(ts).isoformat() if 0 < ts < 2000000000 else datetime.now().isoformat()
    except:
        created = datetime.now().isoformat()
    
    sentiment = str(row.get('review_sentiment', 'neutral')).lower()
    if sentiment == 'positive':
        sp, sn, sneu, sc = 0.8, 0.1, 0.1, 0.75
    elif sentiment == 'negative':
        sp, sn, sneu, sc = 0.1, 0.8, 0.1, -0.75
    else:
        sp, sn, sneu, sc = 0.2, 0.2, 0.6, 0.0
    
    post = {
        'post_id': f"amazon_{asin}_{idx}",
        'source': 'amazon_reviews',
        'subreddit': str(row.get('parent_asin', 'electronics'))[:50],
        'title': str(row.get('title', 'Untitled'))[:500],
        'selftext': str(row.get('text', ''))[:2000],
        'author': str(row.get('user_id', 'anonymous'))[:100],
        'created_utc': created,
        'collected_at': datetime.now().isoformat(),
        'score': int(row.get('rating', 3)) if pd.notna(row.get('rating')) else 3,
        'num_comments': int(row.get('helpful_vote', 0)) if pd.notna(row.get('helpful_vote')) else 0,
        'url': f"https://www.amazon.com/dp/{asin}",
        'permalink': f"/dp/{asin}",
        'sentiment_pos': sp, 'sentiment_neg': sn, 'sentiment_neu': sneu, 'sentiment_compound': sc,
        'sentiment_label': sentiment,
        'embedding': embeddings[idx].tolist()
    }
    posts.append(post)

# Insert in batches
BATCH = 50
success = 0
for i in range(0, len(posts), BATCH):
    batch = posts[i:i+BATCH]
    try:
        supabase.table('reddit_posts').upsert(batch).execute()
        success += len(batch)
        print(f"  Inserted {success}/{len(posts)}")
    except Exception as e:
        print(f"  Error at batch {i//BATCH}: {str(e)[:60]}")

print(f"\n[OK] Total inserted: {success}")

# Verify
result = supabase.table('reddit_posts').select('*', count='exact').execute()
print(f"[VERIFIED] Total in database: {result.count}")

print("\n" + "="*60)
print("COMPLETE!")
print("="*60)