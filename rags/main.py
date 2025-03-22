"""Main entry point for the RAGS system."""

import os
import sys
import argparse
import traceback
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from rags.models import RAGDependencies, create_llm_model
from rags.retriever import search_documents, process_files

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
    parser.add_argument('--llm-url', type=str, default="http://127.0.0.1:8080/v1",
                      help='URL for the LLM server (default: http://127.0.0.1:8080/v1)')
    parser.add_argument('--api-key', type=str, default="not-needed",
                      help='API key for the LLM server (default: not-needed)')
    parser.add_argument('--model-name', type=str, default="gpt-4o",
                      help='Model name to use (default: gpt-4o)')
    
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
        
        # Create LLM model with settings from command line arguments
        local_model = create_llm_model(
            base_url=args.llm_url,
            api_key=args.api_key,
            model_name=args.model_name
        )
        
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
        traceback.print_exc()

def main_cli():
    """Entry point for the CLI command."""
    import asyncio
    asyncio.run(main())

# Allow this module to be imported without running main()
if __name__ == "__main__":
    main_cli()
