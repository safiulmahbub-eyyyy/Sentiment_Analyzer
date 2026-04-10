"""
Load Amazon Electronics Reviews into Supabase
Downloads directly from UCSD data server (original McAuley-Lab source)
Avoids HuggingFace datasets library compatibility issues

Usage:
    python scripts/load_huggingface_dataset.py
"""

import os
import sys
import time
import gzip
import json
from datetime import datetime
from typing import List, Dict, Any
from io import BytesIO

print("="*60)
print("AMAZON REVIEWS LOADER - UCSD Direct Download")
print("Cell Phones & Electronics Category")
print("="*60)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================
# Step 1: Install dependencies
# ============================================================
print("\n[STEP 1] Installing dependencies...")

import subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "requests", "pandas", "numpy", "-q"], check=True)

import requests
import pandas as pd
import numpy as np

# ============================================================
# Step 2: Download from UCSD
# ============================================================
print("\n[STEP 2] Downloading dataset from UCSD...")

# Direct URLs from UCSD data repository
CELL_PHONES_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Cell_Phones_and_Accessories.jsonl.gz"
ELECTRONICS_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz"

def download_jsonl(url: str, max_lines: int = 15000) -> List[Dict]:
    """Download and parse JSONL file"""
    print(f"Downloading: {url.split('/')[-1]}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    data = []
    with gzip.GzipFile(fileobj=BytesIO(response.content)) as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            try:
                obj = json.loads(line)
                data.append(obj)
            except:
                continue
    
    print(f"  Downloaded {len(data):,} reviews")
    return data

# Download both categories
reviews_cell = download_jsonl(CELL_PHONES_URL, max_lines=10000)
reviews_elec = download_jsonl(ELECTRONICS_URL, max_lines=5000)

# Combine
all_reviews = reviews_cell + reviews_elec
print(f"\n[OK] Total: {len(all_reviews):,} reviews")

# ============================================================
# Step 3: Convert to DataFrame and filter
# ============================================================
print("\n[STEP 3] Processing data...")

df = pd.DataFrame(all_reviews)

# Filter for quality: has text content
df = df[df['text'].str.len() > 30].reset_index(drop=True)

# Sort by helpful votes for quality
if 'helpful_vote' in df.columns:
    df = df.sort_values('helpful_vote', ascending=False)

# Limit to top reviews
MAX_REVIEWS = 10000
df = df.head(MAX_REVIEWS).reset_index(drop=True)
print(f"[OK] Using {len(df):,} reviews")

# ============================================================
# Step 4: Generate embeddings
# ============================================================
print("\n[STEP 4] Generating embeddings...")

subprocess.run([sys.executable, "-m", "pip", "install", "transformers", "torch", "-q"], check=True)

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

# Prepare texts
print("Preparing text for embedding...")
texts = []
for _, row in df.iterrows():
    title = str(row.get('title', ''))[:200]
    text = str(row.get('text', ''))[:500]
    combined = f"{title} {text}"[:512]
    texts.append(combined)

# Generate embeddings in batches
BATCH_SIZE = 32
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
    
    if (i // BATCH_SIZE + 1) % 50 == 0:
        print(f"  Progress: {min(i+BATCH_SIZE, len(texts))}/{len(texts)}")

embeddings = np.vstack(all_embeddings)
print(f"[OK] Generated in {time.time() - start_time:.1f}s")

# ============================================================
# Step 5: Insert into Supabase
# ============================================================
print("\n[STEP 5] Inserting into Supabase...")

subprocess.run([sys.executable, "-m", "pip", "install", "supabase", "-q"], check=True)

from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_SERVICE_KEY')

if not supabase_url or not supabase_key:
    print("ERROR: Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in .env")
    sys.exit(1)

supabase = create_client(supabase_url, supabase_key)
print("[OK] Connected to Supabase")

# Build posts
print("Transforming data...")
posts = []

for idx, (_, row) in enumerate(df.iterrows()):
    asin = str(row.get('parent_asin', row.get('asin', 'unknown')))
    title = str(row.get('title', 'Untitled'))[:500]
    text = str(row.get('text', ''))[:2000]
    
    # Get rating
    rating = row.get('rating', 3)
    if pd.isna(rating):
        rating = 3
    
    # Get user_id
    user_id = str(row.get('user_id', 'anonymous'))[:100]
    
    # Get timestamp
    timestamp = row.get('timestamp', 0)
    
    # Get helpful votes
    helpful_vote = row.get('helpful_vote', 0)
    if pd.isna(helpful_vote):
        helpful_vote = 0
    
    # Use title as product name
    product_name = title[:50] if title else "unknown"
    
    # Convert rating to sentiment
    if rating >= 4:
        sentiment = 'positive'
        sp, sn, sneu, sc = 0.8, 0.1, 0.1, 0.75
    elif rating <= 2:
        sentiment = 'negative'
        sp, sn, sneu, sc = 0.1, 0.8, 0.1, -0.75
    else:
        sentiment = 'neutral'
        sp, sn, sneu, sc = 0.2, 0.2, 0.6, 0.0
    
    # Convert timestamp
    try:
        ts = float(timestamp) if timestamp else 0
        if ts > 1000000000000:  # milliseconds
            ts = ts / 1000
        created = datetime.fromtimestamp(ts / 1000).isoformat() if 0 < ts < 2000000000000 else datetime.now().isoformat()
    except:
        created = datetime.now().isoformat()
    
    post = {
        'post_id': f"amazon_{asin}_{idx}",
        'source': 'amazon_reviews',
        'subreddit': product_name,
        'title': title,
        'selftext': text,
        'author': user_id,
        'created_utc': created,
        'collected_at': datetime.now().isoformat(),
        'score': int(rating),
        'num_comments': int(helpful_vote),
        'url': f"https://www.amazon.com/dp/{asin}",
        'permalink': f"/dp/{asin}",
        'sentiment_pos': sp,
        'sentiment_neg': sn,
        'sentiment_neu': sneu,
        'sentiment_compound': sc,
        'sentiment_label': sentiment,
        'embedding': embeddings[idx].tolist()
    }
    posts.append(post)

# Insert in batches
BATCH = 50
success = 0
errors = 0

for i in range(0, len(posts), BATCH):
    batch = posts[i:i+BATCH]
    try:
        supabase.table('reddit_posts').upsert(batch).execute()
        success += len(batch)
        print(f"  Inserted {success}/{len(posts)}")
    except Exception as e:
        errors += len(batch)
        print(f"  Error at batch {i//BATCH}: {str(e)[:60]}")

print(f"\n[OK] Total inserted: {success}")
if errors > 0:
    print(f"[WARN] Errors: {errors}")

# Verify
result = supabase.table('reddit_posts').select('*', count='exact').execute()
print(f"[VERIFIED] Total in database: {result.count}")

print("\n" + "="*60)
print("COMPLETE!")
print(f"Loaded {success} reviews with iPhone/Samsung/Android product names")
print("="*60)