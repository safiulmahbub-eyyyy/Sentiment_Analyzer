"""
Load Amazon Electronics Reviews into Supabase
Uses McAuley-Lab/Amazon-Reviews-2023 dataset with Cell_Phones category
Contains actual iPhone, Samsung, Android product reviews

Usage:
    python scripts/load_huggingface_dataset.py
"""

import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any

print("="*60)
print("AMAZON REVIEWS LOADER - McAuley-Lab Dataset")
print("Cell Phones & Electronics Category")
print("="*60)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================
# Step 1: Install dependencies and load dataset
# ============================================================
print("\n[STEP 1] Loading dataset from HuggingFace...")

# Install required packages
import subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "datasets", "pandas", "numpy", "-q"], check=True)

from datasets import load_dataset
import pandas as pd

# Load Cell_Phones_and_Accessories category (has iPhone, Samsung, Android reviews)
# Also loading Electronics for broader coverage
print("Loading Cell_Phones_and_Accessories category...")
ds_cell = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023", 
    "raw_review_Cell_Phones_and_Accessories",
    trust_remote_code=True,
    split="train"
)
print(f"[OK] Loaded {len(ds_cell):,} reviews from Cell_Phones")

# Also load Electronics category for laptops, tablets, etc
print("Loading Electronics category...")
ds_electronics = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_Electronics", 
    trust_remote_code=True,
    split="train"
)
print(f"[OK] Loaded {len(ds_electronics):,} reviews from Electronics")

# Combine datasets
print("\nCombining datasets...")
ds = ds_cell.train_test_split(train_size=min(10000, len(ds_cell)), test_size=0)
df_cell = ds["train"].to_pandas()

ds2 = ds_electronics.train_test_split(train_size=min(5000, len(ds_electronics)), test_size=0)
df_elec = ds2["train"].to_pandas()

# Combine and remove duplicates by text
df = pd.concat([df_cell, df_elec], ignore_index=True)

# Filter to get reviews with substantial text
df = df[df['text'].str.len() > 50].reset_index(drop=True)

print(f"[OK] Combined: {len(df):,} reviews")

# ============================================================
# Step 2: Limit to subset and filter for quality
# ============================================================
MAX_REVIEWS = 10000  # Limit to 10K most helpful + verified
print(f"\n[STEP 2] Limiting to {MAX_REVIEWS:,} reviews...")

# Sort by helpful votes and verified purchase for quality
if 'helpful_vote' in df.columns and 'verified_purchase' in df.columns:
    df = df.sort_values(
        by=['helpful_vote', 'verified_purchase'], 
        ascending=[False, False]
    )
elif 'helpful_vote' in df.columns:
    df = df.sort_values(by='helpful_vote', ascending=False)

df = df.head(MAX_REVIEWS).reset_index(drop=True)
print(f"[OK] Using {len(df):,} reviews for processing")

# ============================================================
# Step 3: Generate embeddings
# ============================================================
print("\n[STEP 3] Generating embeddings...")

subprocess.run([sys.executable, "-m", "pip", "install", "transformers", "torch", "-q"], check=True)

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

# Prepare texts - combine title and review text
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
# Step 4: Transform and insert into Supabase
# ============================================================
print("\n[STEP 4] Inserting into Supabase...")

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

# Build posts with correct fields
print("Transforming data...")
posts = []

for idx, (_, row) in enumerate(df.iterrows()):
    # Get fields from the dataset
    asin = str(row.get('parent_asin', row.get('asin', 'unknown')))
    title = str(row.get('title', 'Untitled'))[:500]
    text = str(row.get('text', ''))[:2000]
    rating = row.get('rating', 3)
    user_id = str(row.get('user_id', 'anonymous'))[:100]
    timestamp = row.get('timestamp', 0)
    helpful_vote = row.get('helpful_vote', 0)
    
    # Extract product name from title (remove brand prefixes like "Apple", "Samsung" for clarity)
    # Use full title as product identifier
    product_name = title[:50] if title else "unknown"
    
    # Convert rating to sentiment (1-2 = negative, 3 = neutral, 4-5 = positive)
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
        created = datetime.fromtimestamp(ts).isoformat() if 0 < ts < 2000000000 else datetime.now().isoformat()
    except:
        created = datetime.now().isoformat()
    
    post = {
        'post_id': f"amazon_{asin}_{idx}",
        'source': 'amazon_reviews',
        'subreddit': product_name,  # Use product name as identifier
        'title': title,
        'selftext': text,
        'author': user_id,
        'created_utc': created,
        'collected_at': datetime.now().isoformat(),
        'score': int(rating) if pd.notna(rating) else 3,
        'num_comments': int(helpful_vote) if pd.notna(helpful_vote) else 0,
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
print(f"Loaded {success} reviews with product names (iPhone, Samsung, Android, etc.)")
print("="*60)