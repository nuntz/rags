"""Document retrieval functionality for the RAGS system."""

from pathlib import Path
import os
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple

from pydantic_ai import RunContext

from rags.models import RAGDependencies
from rags.utils import (
    calculate_file_hash, 
    split_markdown_into_chunks, 
    load_file_registry, 
    save_file_registry
)

async def search_documents(ctx: RunContext[RAGDependencies], query: str, n_results: int = 5) -> str:
    """
    Search for documents related to the query using vector similarity.
    
    Args:
        query: The user's question or search query
        n_results: Number of relevant documents to retrieve (default: 5)
    
    Returns:
        String containing the most relevant document snippets
    """
    # Get the collection
    collection = ctx.deps.chroma_client.get_collection(name=ctx.deps.collection_name)
    
    # Generate embedding for the query using the embedding model
    query_embedding = ctx.deps.embedding_model.encode(query).tolist()
    
    # Search the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format the results
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    
    if not documents:
        return "No relevant documents found."
    
    # Format the retrieved contexts with source information
    formatted_results = []
    for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
        source = metadata.get("source", "Unknown")
        header = metadata.get("header", "")
        similarity = 1 - distance  # Convert distance to similarity score
        formatted_results.append(f"[Document {i+1}] (Source: {source}, Section: {header}, Relevance: {similarity:.2f})\n{doc}\n")
    
    return "\n".join(formatted_results)

async def process_files(docs_path, collection_name, embedding_model, chroma_client, db_path, force_reload=False):
    # Setup registry to track file hashes
    data_dir = Path(db_path).parent / "data"
    data_dir.mkdir(exist_ok=True)
    registry_path = data_dir / "file_registry.json"
    
    registry = load_file_registry(registry_path)
    
    # Create or get the collection
    try:
        if force_reload:
            try:
                chroma_client.delete_collection(collection_name)
                print(f"Deleted existing collection '{collection_name}' due to force reload")
            except:
                pass
            collection = chroma_client.create_collection(name=collection_name)
            # Clear registry if we're force reloading
            registry = {}
        else:
            try:
                collection = chroma_client.get_collection(name=collection_name)
                print(f"Using existing collection '{collection_name}'")
            except:
                collection = chroma_client.create_collection(name=collection_name)
                print(f"Created new collection '{collection_name}'")
    except Exception as e:
        print(f"Error with collection: {e}")
        return False
    
    # Get all markdown files (recursively through all subfolders)
    markdown_files = []
    docs_path = Path(docs_path)
    if docs_path.is_dir():
        # The ** pattern ensures we search through all subdirectories recursively
        markdown_files = list(docs_path.glob("**/*.md"))
        print(f"Found {len(markdown_files)} markdown files in {docs_path} and its subdirectories")
    elif docs_path.suffix.lower() == '.md':
        markdown_files = [docs_path]
        print(f"Processing single markdown file: {docs_path}")
    else:
        print(f"Invalid documentation path: {docs_path}")
        return False
    
    # Check if any files need processing
    files_to_process = []
    for file_path in markdown_files:
        file_hash = calculate_file_hash(file_path)
        try:
            rel_path = str(file_path.relative_to(Path.cwd())) if file_path.is_absolute() else str(file_path)
        except ValueError:
            # If the file is not in a subpath of the current directory, use the full path
            rel_path = str(file_path)
        
        if rel_path not in registry or registry[rel_path]["hash"] != file_hash:
            files_to_process.append(file_path)
            # Update registry
            registry[rel_path] = {
                "hash": file_hash,
                "last_processed": datetime.now().isoformat()
            }
    
    if not files_to_process and not force_reload:
        print("All files are up to date in the database.")
        save_file_registry(registry, registry_path)
        return True
    
    # Load and process documents
    all_chunks = []
    all_embeddings = []
    all_metadatas = []
    all_ids = []
    
    chunk_id = 0
    
    for file_path in files_to_process:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            try:
                rel_path = str(file_path.relative_to(Path.cwd())) if file_path.is_absolute() else str(file_path)
            except ValueError:
                # If the file is not in a subpath of the current directory, use the full path
                rel_path = str(file_path)
            print(f"Processing {rel_path}...")
            chunks = split_markdown_into_chunks(content, rel_path)
            
            for chunk_text, metadata in chunks:
                all_chunks.append(chunk_text)
                all_metadatas.append(metadata)
                all_ids.append(f"chunk_{chunk_id}")
                chunk_id += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Generate embeddings for all chunks
    if all_chunks:
        all_embeddings = embedding_model.encode(all_chunks).tolist()
        
        # Add to collection
        collection.add(
            embeddings=all_embeddings,
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )
        print(f"Added {len(all_chunks)} chunks to the vector database")
    
    # Save updated registry
    save_file_registry(registry, registry_path)
    return True
