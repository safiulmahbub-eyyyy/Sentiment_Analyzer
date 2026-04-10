"""
Configuration for embedding models
"""

# Embedding model settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Batch sizes
EMBEDDING_BATCH_SIZE = 32  # Number of texts to embed at once
UPDATE_BATCH_SIZE = 100    # Number of embeddings to update at once

# Processing settings
MAX_TEXT_LENGTH = 512  # Maximum characters to use for embedding
COMBINE_TITLE_BODY = True  # Combine title and selftext for embedding

# Model loading settings
DEVICE = "cpu"  # Use "cuda" if GPU available, "cpu" otherwise
SHOW_PROGRESS = True  # Show progress bars during processing
