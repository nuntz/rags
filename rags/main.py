import os
import sys
import argparse
import hashlib
from datetime import datetime
import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

# Define our dependencies
@dataclass
class RAGDependencies:
    chroma_client: chromadb.Client
    collection_name: str
    embedding_model: SentenceTransformer

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

# Create a model with a custom provider pointing to a local server
def create_llm_model(base_url="http://127.0.0.1:8080/v1", api_key="not-needed", model_name="gpt-4o"):
    local_provider = OpenAIProvider(
        base_url=base_url,
        api_key=api_key
    )
    
    return OpenAIModel(
        model_name,
        provider=local_provider
    )

# Create the search tool function - we'll register it with the agent later
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
        rel_path = str(file_path.relative_to(Path.cwd()) if file_path.is_absolute() else file_path)
        
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
                
            rel_path = str(file_path.relative_to(Path.cwd()) if file_path.is_absolute() else file_path)
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

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RAGS - Retrieval-Augmented Generation System')
    parser.add_argument('docs_path', type=str,
                      help='Path to the documentation directory or a specific markdown file')
    parser.add_argument('--force-reload', action='store_true',
                      help='Force reprocessing of all files')
    parser.add_argument('--collection-name', type=str, default='default_collection',
                      help='Name of the Chroma collection to use')
    parser.add_argument('--db-path', type=str, default=os.path.expanduser('~/.local/share/rags/chroma_db'),
                      help='Path to the ChromaDB database directory')
    parser.add_argument('--request-limit', type=int, default=50,
                      help='Maximum number of requests to make to the LLM (default: 50)')
    
    args = parser.parse_args()
    
    try:
        # Initialize components
        db_path = args.db_path
        os.makedirs(db_path, exist_ok=True)
        
        print(f"Using ChromaDB at: {db_path}")
        chroma_client = chromadb.PersistentClient(path=db_path)
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Process files
        success = await process_files(
            args.docs_path, 
            args.collection_name, 
            embedding_model, 
            chroma_client, 
            db_path,
            args.force_reload
        )
        
        if not success:
            print("Error processing documentation files.")
            return
        
        # Create LLM model with default settings
        local_model = create_llm_model()
        
        # Create agent
        rag_agent = Agent(
            local_model,
            deps_type=RAGDependencies,
            model_settings={'timeout': 60.0},
            system_prompt=(
                "You are a helpful assistant that answers questions based on retrieved documents. "
                "If you don't know the answer based on the retrieved context, say so - don't make things up."
            ),
        )
        
        # Register the search tool with the agent here, not at the module level
        search_tool = rag_agent.tool(search_documents)
        
        # Set up dependencies for the agent
        deps = RAGDependencies(
            chroma_client=chroma_client,
            collection_name=args.collection_name,
            embedding_model=embedding_model
        )
        
        print(f"\nRAG system ready! Using documents from: {args.docs_path}")
        print(f"Collection: {args.collection_name}")
        print("Type 'exit' or 'quit' to end the session.\n")
        
        # Interactive query loop
        while True:
            query = input("\nEnter your question: ")
            
            if query.lower() in ('exit', 'quit'):
                break
                
            print("\nQuerying the agent using local LLM server...")
            # Run the agent with a user query
            result = await rag_agent.run(
                query, 
                deps=deps,
                usage_limits=UsageLimits(request_limit=args.request_limit)
            )
            print("\nAnswer:")
            print(result.data)
        
    except Exception as e:
        print(f"An error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

def main_cli():
    """Entry point for the CLI command."""
    import asyncio
    asyncio.run(main())

# Allow this module to be imported without running main()
if __name__ == "__main__":
    main_cli()
