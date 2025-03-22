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
            # If we have content from previous section, add it as a chunk
            if current_chunk:
                metadata = {
                    "source": str(filename),
                    "header": current_header
                }
                chunks.append((current_chunk.strip(), metadata))
                current_chunk = ""
                
            current_header = section.strip()
            continue
            
        # If we have content, process it
        if section.strip():
            # Split the content into paragraphs
            paragraphs = section.strip().split('\n\n')
            
            for paragraph in paragraphs:
                if not paragraph.strip():
                    continue
                
                # Check if current chunk plus this paragraph would exceed max size
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
                    # Add paragraph to current chunk
                    if current_chunk:
                        current_chunk += " " + paragraph
                    else:
                        current_chunk = paragraph
                
                # Check if the current chunk is already too large
                # This handles cases where a single paragraph exceeds max_chunk_size
                while len(current_chunk) > max_chunk_size:
                    # Take the first max_chunk_size characters (approximately)
                    words = current_chunk.split()
                    chunk_words = []
                    chunk_length = 0
                    
                    for word in words:
                        if chunk_length + len(word) + 1 > max_chunk_size:
                            break
                        chunk_words.append(word)
                        chunk_length += len(word) + 1  # +1 for space
                    
                    if not chunk_words:  # If a single word is too long
                        chunk_words = [words[0]]
                    
                    chunk_text = ' '.join(chunk_words)
                    metadata = {
                        "source": str(filename),
                        "header": current_header
                    }
                    chunks.append((chunk_text.strip(), metadata))
                    
                    # Remove the processed part from current_chunk with overlap
                    overlap_words = chunk_words[-int(overlap/5):] if len(chunk_words) > int(overlap/5) else chunk_words
                    overlap_text = ' '.join(overlap_words)
                    remaining_text = current_chunk[len(chunk_text):]
                    current_chunk = overlap_text + remaining_text
    
    # Don't forget to add the last chunk
    if current_chunk:
        metadata = {
            "source": str(filename),
            "header": current_header
        }
        chunks.append((current_chunk.strip(), metadata))
    
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
