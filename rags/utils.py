"""Utility functions for the RAGS system."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
import re
from typing import List, Dict, Tuple, Any

# Calculate file hash to detect changes
def calculate_file_hash(file_path):
    with open(file_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash

# Function to split text into chunks
def split_markdown_into_chunks(text, filename, max_chunk_size=1000, overlap=200):
    """Split text into chunks of maximum size with overlap between chunks."""
    if not text.strip():
        return []
    
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        # Add word length plus space
        word_length = len(word) + 1
        
        if current_length + word_length > max_chunk_size and current_chunk:
            # Create chunk with current words
            chunk_text = ' '.join(current_chunk)
            metadata = {"source": str(filename), "header": ""}
            chunks.append((chunk_text.strip(), metadata))
            
            # Start new chunk with overlap
            overlap_size = min(len(current_chunk), int(overlap/5))
            current_chunk = current_chunk[-overlap_size:] if overlap_size > 0 else []
            current_length = sum(len(w) + 1 for w in current_chunk)
        
        current_chunk.append(word)
        current_length += word_length
    
    # Add the last chunk if there's anything left
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        metadata = {"source": str(filename), "header": ""}
        chunks.append((chunk_text.strip(), metadata))
    
    return chunks

# Function to track processed files and their hashes
def load_file_registry(registry_path):
    if registry_path.exists():
        with open(registry_path, 'r') as f:
            return json.load(f)
    return {}

def save_file_registry(registry, registry_path):
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
