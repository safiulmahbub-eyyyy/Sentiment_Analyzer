"""
Load Amazon Electronics Reviews into Supabase
Uses bagadbilla/amazon-reviews-2023-trimmed dataset (Parquet format - works with Python 3.14)
Contains Cell_Phones and Electronics categories with iPhone, Samsung, Android reviews

Usage:
    python scripts/load_huggingface_dataset.py
"""

import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any

print("="*60)
print("AMAZON REVIEWS LOADER")
print("bagadbilla/amazon-reviews-2023-trimmed Dataset")
print("="*60)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================
# Step 1: Install dependencies
# ============================================================
print("\n[STEP 1] Installing dependencies...")

import subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "datasets", "pandas", "numpy", "transformers", "torch", "supabase", "python-dotenv", "-q"], check=True)

from datasets import load_dataset
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# ============================================================
# Step 2: Load dataset from HuggingFace
# ============================================================
print("\n[STEP 2] Loading dataset from HuggingFace...")

# Load Cell_Phones_and_Accessories (has iPhone, Samsung, Android reviews)
print("Loading Cell_Phones_and_Accessories...")
ds_cell = load_dataset(
    "bagadbilla/amazon-reviews-2023-trimmed",
    data_dir="Cell_Phones_and_Accessories",
    split="train"
)
print(f"[OK] Loaded {len(ds_cell):,} reviews")

# Also load Electronics for laptops, tablets
print("Loading Electronics...")
ds_electronics = load_dataset(
    "bagadbilla/amazon-reviews-2023-trimmed",
    data_dir="Electronics",
    split="train"
)
print(f"[OK] Loaded {len(ds_electronics):,} reviews")

# ============================================================
# Step 3: Process and filter
# ============================================================
print("\n[STEP 3] Processing data...")

# Convert to DataFrame
df_cell = ds_cell.to_pandas()
df_elec = ds_electronics.to_pandas()

# Combine
df = pd.concat([df_cell, df_elec], ignore_index=True)

# Filter for quality - has text content
df = df[df['text'].str.len() > 30].reset_index(drop=True)

# Sort by rating (higher = more helpful reviews often)
df = df.sort_values('rating', ascending=False)

# Limit to top reviews
MAX_REVIEWS = 10000
df = df.head(MAX_REVIEWS).reset_index(drop=True)
print(f"[OK] Using {len(df):,} reviews")

# ============================================================
# Step 4: Generate embeddings
# ============================================================
print("\n[STEP 4] Generating embeddings...")

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

# Generate embeddings
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
    # Get fields - this dataset has rating, title, text
    asin = f"asin_{idx}"  # No asin in this dataset, create placeholder
    title = str(row.get('title', 'Untitled'))[:500]
    text = str(row.get('text', ''))[:2000]
    
    rating = row.get('rating', 3)
    if pd.isna(rating):
        rating = 3
    
    # Use product title as identifier
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
    
    post = {
        'post_id': f"amazon_{asin}_{idx}",
        'source': 'amazon_reviews',
        'subreddit': product_name,
        'title': title,
        'selftext': text,
        'author': 'anonymous',
        'created_utc': datetime.now().isoformat(),
        'collected_at': datetime.now().isoformat(),
        'score': int(rating),
        'num_comments': 0,
        'url': 'https://www.amazon.com',
        'permalink': '/',
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
print(f"Loaded {success} reviews with iPhone/Samsung/Android")
print("="*60)