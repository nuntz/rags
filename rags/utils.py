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
    # Split by headers
    header_pattern = r'(#{1,6}\s+.+)'
    sections = re.split(header_pattern, text)
    
    chunks = []
    current_header = ""
    current_chunk = ""
    
    for i, section in enumerate(sections):
        # Check if this section is a header
        if i % 2 == 1:
            current_header = section.strip()
            continue
            
        # If we have content, process it
        if section.strip():
            # Split the content into paragraphs
            paragraphs = section.strip().split('\n\n')
            
            for paragraph in paragraphs:
                if not paragraph.strip():
                    continue
                    
                # If adding this paragraph would make the chunk too large, save the current chunk
                if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                    metadata = {
                        "source": str(filename),
                        "header": current_header
                    }
                    chunks.append((current_chunk.strip(), metadata))
                    
                    # Start a new chunk with overlap
                    words = current_chunk.split()
                    overlap_text = ' '.join(words[-int(overlap/5):]) if len(words) > int(overlap/5) else current_chunk
                    current_chunk = overlap_text + " " + paragraph
                else:
                    if current_chunk:
                        current_chunk += " " + paragraph
                    else:
                        current_chunk = paragraph
        
        # Don't forget to add the last chunk
        if current_chunk:
            metadata = {
                "source": str(filename),
                "header": current_header
            }
            chunks.append((current_chunk.strip(), metadata))
            current_chunk = ""
    
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
