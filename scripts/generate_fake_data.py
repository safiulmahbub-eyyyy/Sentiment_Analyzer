"""
Generate Fake Amazon Reviews Dataset
Creates 10K fake reviews with embeddings for iPhone, Samsung, Android, Laptop products
Then inserts into Supabase

Usage:
    python scripts/generate_fake_data.py
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta
import random

print("="*60)
print("GENERATING FAKE AMAZON REVIEWS DATASET")
print("="*60)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================
# Step 1: Install dependencies
# ============================================================
print("\n[STEP 1] Installing dependencies...")

import subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "transformers", "torch", "supabase", "python-dotenv", "-q"], check=True)

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# ============================================================
# Step 2: Define fake reviews data
# ============================================================
print("\n[STEP 2] Creating fake review data...")

# Product definitions with realistic reviews
PRODUCTS = {
    "iPhone 15 Pro": [
        ("Amazing battery life!", "Battery lasts all day even with heavy use. Best iPhone I've ever owned."),
        ("Great camera", "Camera quality is incredible for professional photos."),
        ("Premium build", "Titanium frame feels premium and sturdy."),
        ("Fast performance", "A17 chip handles everything smoothly."),
        ("Good but pricey", "Great phone but expensive. Worth the upgrade?"),
        ("Screen issues", "Display is great but scratched easily."),
        ("Best iPhone ever", "Upgraded from 13 and loving the improvements!"),
        ("Not impressed", "Expected more improvements over 14 Pro."),
    ],
    "iPhone 15": [
        ("Great everyday phone", "Perfect for daily use without breaking bank."),
        ("Good value", "Best iPhone for the money in 2024."),
        ("Solid performer", "Handles all my apps and games fine."),
        ("Camera is good", "Photos look great for social media."),
        ("Missing features", "No telephoto lens is disappointing."),
    ],
    "Samsung Galaxy S24": [
        ("Best Android phone", "Finally switched from iPhone and love it!"),
        ("Galaxy AI is cool", "Circle to search is incredibly useful."),
        ("Great display", "OLED screen is beautiful."),
        ("Good battery", "Lasts all day with heavy usage."),
        ("Not worth upgrade", "Similar to S23, not worth the price."),
    ],
    "Samsung Galaxy S24 Ultra": [
        ("S Pen is amazing", "Perfect fornote-taking and editing."),
        ("Best Android flagship", "Premium build and features."),
        ("Expensive but worth", "Cheapest flagship with all features."),
        ("Camera king", "200MP camera is incredible."),
        ("Too big", "Phablet size not for everyone."),
    ],
    "Google Pixel 8": [
        ("Best camera phone", "Computational photography is mind-blowing!"),
        ("AI features rock", "Magic eraser and best take are great."),
        ("Clean software", "Pure Android experience is nice."),
        ("Good value", "Best flagship value this year."),
        ("Average battery", "Battery could be better."),
    ],
    "Google Pixel 8 Pro": [
        ("Pro camera features", "Manual controls are great for pros."),
        ("Best Android", "Simply the best Android experience."),
        ("Tensor chip is good", "AI features work surprisingly well."),
        ("Expensive", "Pricey but feature-rich."),
    ],
    "MacBook Pro 14": [
        ("Workstation power", "M3 Pro handles video editing perfectly."),
        ("Best laptop ever", "Build quality is unmatched."),
        ("Battery lasts forever", "20+ hours battery life is real!"),
        ("Perfect for development", "Compiles code incredibly fast."),
        ("Too expensive", "Price is hard to justify."),
    ],
    "MacBook Air M3": [
        ("Perfect daily laptop", "Great for work and casual use."),
        ("Best value Mac", "M3 is plenty powerful."),
        ("Silent operation", "Fanless design works great."),
        ("Not for pros", "Needs Pro chip for heavy workloads."),
    ],
    "Dell XPS 15": [
        ("Beautiful display", "OLED screen is stunning."),
        ("Great for development", "Powerful and portable."),
        ("Premium design", "Best Windows laptop hands down."),
        ("Expensive", "Pricey but worth it."),
        ("Heating issues", "Runs warm under load."),
    ],
    "Lenovo ThinkPad X1": [
        ("Business standard", "Best business laptop."),
        ("Incredible keyboard", "The best keyboard on any laptop."),
        ("Reliable", "Built like a tank."),
        ("Expensive", "Not for budget users."),
    ],
    "ASUS ROG Laptop": [
        ("Gaming beast", "Handles any game at max settings!"),
        ("Great display", "High refresh rate is smooth."),
        ("Heavy but powerful", "Desktop performance in laptop."),
        ("Expensive", "Gaming laptops are pricey."),
    ],
    "AirPods Pro 2": [
        ("Best earbuds", "Noise cancellation is incredible."),
        ("Perfect fit", "Finally got the right ear tips."),
        ("Sound quality great", "Audiophile-quality sound."),
        ("Overrated", "Not worth the premium price."),
    ],
    "Samsung Galaxy Buds": [
        ("Great alternative", "Best Galaxy phone companion."),
        ("Good ANC", "Noise cancellation works well."),
        ("Comfortable", "Fit well for long sessions."),
    ],
}

# Sentiment distributions
SENTIMENTS = {
    "positive": (0.7, ["Amazing", "Great", "Best", "Love", "Perfect", "Incredible", "Excellent", "Worth"]),
    "neutral": (0.2, ["Okay", "Average", "Decent", "Fine", "Normal", "Standard"]),
    "negative": (0.1, ["Disappointing", "Poor", "Bad", "Overrated", "Not worth", "Avoid"]),
}

def generate_review_text(product: str, sentiment: str) -> tuple:
    """Generate a random review"""
    base_reviews = PRODUCTS.get(product, [(f"Great {product}", f"Really like this {product}")])
    
    base_title, base_text = random.choice(base_reviews)
    
    # Modify based on sentiment
    if sentiment == "positive":
        modifier = random.choice(SENTIMENTS["positive"][1])
        title = f"{modifier} {product}!"
    elif sentiment == "negative":
        modifier = random.choice(SENTIMENTS["negative"][1])
        title = f"{modifier} {product}"
    else:
        modifier = random.choice(SENTIMENTS["neutral"][1])
        title = f"{modifier} {product}"
    
    # Expand text
    sentiments_phrase = {
        "positive": "I highly recommend this product!",
        "neutral": "It's an okay product for the price.",
        "negative": "I would not recommend this.",
    }
    
    text = f"{base_text} {sentiments_phrase[sentiment]}"
    
    return title, text

# Generate reviews
NUM_REVIEWS = 10000
print(f"Generating {NUM_REVIEWS} reviews...")

reviews = []
for i in range(NUM_REVIEWS):
    product = random.choice(list(PRODUCTS.keys()))
    
    # Determine sentiment with distribution
    rand = random.random()
    if rand < 0.7:
        sentiment = "positive"
    elif rand < 0.9:
        sentiment = "neutral"
    else:
        sentiment = "negative"
    
    title, text = generate_review_text(product, sentiment)
    
    # Generate random timestamp in last 2 years
    days_ago = random.randint(0, 730)
    created = datetime.now() - timedelta(days=days_ago)
    
    review = {
        "product": product,
        "title": title,
        "text": text,
        "sentiment": sentiment,
        "rating": 5 if sentiment == "positive" else (3 if sentiment == "neutral" else 2),
        "created_utc": created.isoformat(),
        "helpful_votes": random.randint(0, 100) if sentiment == "positive" else random.randint(0, 20),
    }
    reviews.append(review)

print(f"[OK] Generated {len(reviews)} fake reviews")

# ============================================================
# Step 3: Generate embeddings
# ============================================================
print("\n[STEP 3] Generating embeddings...")

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Prepare texts for embedding
texts = [f"{r['title']} {r['text']}"[:512] for r in reviews]

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
# Step 4: Insert into Supabase
# ============================================================
print("\n[STEP 4] Inserting into Supabase...")

from supabase import create_client
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_SERVICE_KEY')

if not supabase_url or not supabase_key:
    print("ERROR: Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in .env")
    print("Please check your .env file")
    sys.exit(1)

supabase = create_client(supabase_url, supabase_key)
print("[OK] Connected to Supabase")

# Transform and insert
print("Transforming and inserting data...")
posts = []

for idx, (review, embedding) in enumerate(zip(reviews, embeddings)):
    sentiment = review['sentiment']
    rating = review['rating']
    
    if sentiment == 'positive':
        sp, sn, sneu, sc = 0.8, 0.1, 0.1, 0.75
    elif sentiment == 'negative':
        sp, sn, sneu, sc = 0.1, 0.8, 0.1, -0.75
    else:
        sp, sn, sneu, sc = 0.2, 0.2, 0.6, 0.0
    
    post = {
        'post_id': f"fake_{review['product'].replace(' ', '_')}_{idx}",
        'source': 'amazon_reviews',
        'subreddit': review['product'],
        'title': review['title'][:500],
        'selftext': review['text'][:2000],
        'author': f"user_{random.randint(1000, 9999)}",
        'created_utc': review['created_utc'],
        'collected_at': datetime.now().isoformat(),
        'score': review['rating'],
        'num_comments': review['helpful_votes'],
        'url': f"https://www.amazon.com/{review['product'].replace(' ', '-').lower()}",
        'permalink': f"/dp/{random.randint(100000, 999999)}",
        'sentiment_pos': sp,
        'sentiment_neg': sn,
        'sentiment_neu': sneu,
        'sentiment_compound': sc,
        'sentiment_label': sentiment,
        'embedding': embedding.tolist()
    }
    posts.append(post)

# Insert in batches
BATCH = 50
success = 0
errors = 0

print(f"Inserting {len(posts)} records...")
for i in range(0, len(posts), BATCH):
    batch = posts[i:i+BATCH]
    try:
        supabase.table('reddit_posts').upsert(batch).execute()
        success += len(batch)
        print(f"  Inserted {success}/{len(posts)}")
    except Exception as e:
        errors += len(batch)
        print(f"  Error at batch {i//BATCH}: {str(e)[:80]}")

print(f"\n[OK] Total inserted: {success}")
if errors > 0:
    print(f"[WARN] Errors: {errors}")

# Verify
result = supabase.table('reddit_posts').select('*', count='exact').execute()
print(f"[VERIFIED] Total in database: {result.count}")

# Show sample products
print("\n[DATA SUMMARY]")
print(f"Total reviews: {len(posts)}")
products_count = {}
for post in posts:
    p = post['subreddit']
    products_count[p] = products_count.get(p, 0) + 1

for product, count in sorted(products_count.items(), key=lambda x: -x[1]):
    print(f"  {product}: {count} reviews")

print("\n" + "="*60)
print("COMPLETE!")
print(f"Generated {success} fake reviews with embeddings")
print("Products: iPhone, Samsung, Android, MacBook, Dell, AirPods")
print("="*60)