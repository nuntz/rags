"""Tests for the document retrieval functionality."""

import os
import tempfile
import pytest
from pathlib import Path
import json
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import chromadb
import numpy as np

from unittest.mock import MagicMock, patch, AsyncMock, create_autospec
from pydantic_ai import RunContext
import pytest_asyncio

from rags.models import RAGDependencies
from rags.retriever import search_documents, process_files


class TestSearchDocuments:
    """Tests for the search_documents function."""

    @pytest.fixture
    def mock_deps(self):
        """Create mock dependencies for testing."""
        deps = MagicMock(spec=RAGDependencies)
        deps.collection_name = "test_collection"
        deps.chroma_client = MagicMock()
        deps.embedding_model = MagicMock()
        return deps

    @pytest.fixture
    def mock_context(self, mock_deps):
        """Create a mock RunContext with dependencies."""
        # Create a mock RunContext with all required parameters
        mock_context = create_autospec(RunContext)
        # Set the deps attribute directly
        mock_context.deps = mock_deps
        return mock_context

    @pytest.mark.asyncio
    async def test_empty_collection(self, mock_context):
        """Test search_documents with an empty collection."""
        # Setup mock collection with no results
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        mock_context.deps.chroma_client.get_collection.return_value = mock_collection
        mock_context.deps.embedding_model.encode.return_value = np.array([0.1, 0.2, 0.3])

        # Execute search
        result = await search_documents(mock_context, "test query")
        
        # Verify results
        assert "No relevant documents found." in result
        mock_context.deps.chroma_client.get_collection.assert_called_once_with(
            name=mock_context.deps.collection_name
        )

    @pytest.mark.asyncio
    async def test_with_known_documents(self, mock_context):
        """Test search_documents with known documents and predictable queries."""
        # Setup mock collection with known results
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [[
                "This is document 1 content.",
                "This is document 2 content."
            ]],
            "metadatas": [[
                {"source": "doc1.md", "header": "Header 1"},
                {"source": "doc2.md", "header": "Header 2"}
            ]],
            "distances": [[0.1, 0.3]]
        }
        mock_context.deps.chroma_client.get_collection.return_value = mock_collection
        mock_context.deps.embedding_model.encode.return_value = np.array([0.1, 0.2, 0.3])

        # Execute search
        result = await search_documents(mock_context, "test query")
        
        # Verify results
        assert "Document 1" in result
        assert "Document 2" in result
        assert "doc1.md" in result
        assert "doc2.md" in result
        assert "Header 1" in result
        assert "Header 2" in result
        assert "Relevance: 0.90" in result  # 1 - 0.1
        assert "Relevance: 0.70" in result  # 1 - 0.3

    @pytest.mark.asyncio
    async def test_varying_n_results(self, mock_context):
        """Test with varying numbers of results (n_results parameter)."""
        # Setup mock collection
        mock_collection = MagicMock()
        mock_context.deps.chroma_client.get_collection.return_value = mock_collection
        mock_context.deps.embedding_model.encode.return_value = np.array([0.1, 0.2, 0.3])

        # Test with n_results=3
        await search_documents(mock_context, "test query", n_results=3)
        mock_collection.query.assert_called_with(
            query_embeddings=[mock_context.deps.embedding_model.encode.return_value.tolist()],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )

        # Test with n_results=10
        mock_collection.reset_mock()
        await search_documents(mock_context, "test query", n_results=10)
        mock_collection.query.assert_called_with(
            query_embeddings=[mock_context.deps.embedding_model.encode.return_value.tolist()],
            n_results=10,
            include=["documents", "metadatas", "distances"]
        )

    @pytest.mark.asyncio
    async def test_edge_case_similarity_scores(self, mock_context):
        """Test with edge case similarity scores."""
        # Setup mock collection with edge case distances
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [[
                "Perfect match document.",
                "Worst match document.",
                "Middle match document."
            ]],
            "metadatas": [[
                {"source": "perfect.md", "header": "Perfect"},
                {"source": "worst.md", "header": "Worst"},
                {"source": "middle.md", "header": "Middle"}
            ]],
            "distances": [[0.0, 1.0, 0.5]]  # Perfect match, worst match, middle match
        }
        mock_context.deps.chroma_client.get_collection.return_value = mock_collection
        mock_context.deps.embedding_model.encode.return_value = np.array([0.1, 0.2, 0.3])

        # Execute search
        result = await search_documents(mock_context, "test query")
        
        # Verify results
        assert "Relevance: 1.00" in result  # 1 - 0.0
        assert "Relevance: 0.00" in result  # 1 - 1.0
        assert "Relevance: 0.50" in result  # 1 - 0.5


class TestProcessFiles:
    """Tests for the process_files function."""
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        model = MagicMock()
        model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        return model
    
    @pytest.fixture
    def mock_chroma_client(self):
        """Create a mock ChromaDB client."""
        client = MagicMock()
        collection = MagicMock()
        client.get_collection.return_value = collection
        client.create_collection.return_value = collection
        return client

    @pytest.mark.asyncio
    async def test_empty_directory(self, mock_embedding_model, mock_chroma_client):
        """Test process_files with an empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "db")
            
            # Call process_files with empty directory
            result = await process_files(
                temp_dir, 
                "test_collection",
                mock_embedding_model,
                mock_chroma_client,
                db_path
            )
            
            # Verify results
            assert result is True
            # Collection should be created but no documents added
            mock_chroma_client.create_collection.assert_called_once()
            collection = mock_chroma_client.create_collection.return_value
            collection.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_markdown_file(self, mock_embedding_model, mock_chroma_client):
        """Test with a single markdown file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test markdown file
            file_path = os.path.join(temp_dir, "test.md")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("# Test Header\n\nThis is test content.")
            
            db_path = os.path.join(temp_dir, "db")
            
            # Mock the embedding model to return appropriate shape
            mock_embedding_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
            
            # Call process_files
            result = await process_files(
                file_path,  # Pass the file path directly
                "test_collection",
                mock_embedding_model,
                mock_chroma_client,
                db_path
            )
            
            # Verify results
            assert result is True
            collection = mock_chroma_client.create_collection.return_value
            collection.add.assert_called_once()
            # Check that the registry was created
            registry_path = os.path.join(temp_dir, "data", "file_registry.json")
            assert os.path.exists(registry_path)

    @pytest.mark.asyncio
    async def test_multiple_markdown_files(self, mock_embedding_model, mock_chroma_client):
        """Test with directory containing multiple markdown files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple test markdown files
            for i in range(3):
                file_path = os.path.join(temp_dir, f"test{i}.md")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Test Header {i}\n\nThis is test content {i}.")
            
            db_path = os.path.join(temp_dir, "db")
            
            # Mock the embedding model to return appropriate shape
            mock_embedding_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
            
            # Call process_files
            result = await process_files(
                temp_dir,
                "test_collection",
                mock_embedding_model,
                mock_chroma_client,
                db_path
            )
            
            # Verify results
            assert result is True
            collection = mock_chroma_client.create_collection.return_value
            collection.add.assert_called_once()
            # Verify registry contains all files
            registry_path = os.path.join(temp_dir, "data", "file_registry.json")
            with open(registry_path, 'r') as f:
                registry = json.load(f)
                assert len(registry) == 3

    @pytest.mark.asyncio
    async def test_nested_directory_structure(self, mock_embedding_model, mock_chroma_client):
        """Test with nested directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested directory structure
            nested_dir = os.path.join(temp_dir, "nested")
            os.makedirs(nested_dir)
            
            # Create files in both root and nested directories
            root_file = os.path.join(temp_dir, "root.md")
            with open(root_file, 'w', encoding='utf-8') as f:
                f.write("# Root File\n\nThis is root content.")
                
            nested_file = os.path.join(nested_dir, "nested.md")
            with open(nested_file, 'w', encoding='utf-8') as f:
                f.write("# Nested File\n\nThis is nested content.")
            
            db_path = os.path.join(temp_dir, "db")
            
            # Mock the embedding model to return appropriate shape
            mock_embedding_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            
            # Call process_files
            result = await process_files(
                temp_dir,
                "test_collection",
                mock_embedding_model,
                mock_chroma_client,
                db_path
            )
            
            # Verify results
            assert result is True
            collection = mock_chroma_client.create_collection.return_value
            collection.add.assert_called_once()
            # Verify registry contains both files
            registry_path = os.path.join(temp_dir, "data", "file_registry.json")
            with open(registry_path, 'r') as f:
                registry = json.load(f)
                assert len(registry) == 2

    @pytest.mark.asyncio
    async def test_force_reload(self, mock_embedding_model, mock_chroma_client):
        """Test force_reload=True vs. force_reload=False behavior."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test markdown file
            file_path = os.path.join(temp_dir, "test.md")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("# Test Header\n\nThis is test content.")
            
            db_path = os.path.join(temp_dir, "db")
            
            # First call with force_reload=False
            await process_files(
                temp_dir,
                "test_collection",
                mock_embedding_model,
                mock_chroma_client,
                db_path,
                force_reload=False
            )
            
            # Reset mock to check calls in second run
            mock_chroma_client.reset_mock()
            collection = MagicMock()
            mock_chroma_client.get_collection.return_value = collection
            mock_chroma_client.create_collection.return_value = collection
            
            # Second call with force_reload=True
            await process_files(
                temp_dir,
                "test_collection",
                mock_embedding_model,
                mock_chroma_client,
                db_path,
                force_reload=True
            )
            
            # Verify delete_collection was called
            mock_chroma_client.delete_collection.assert_called_once_with("test_collection")
            # Verify create_collection was called (not get_collection)
            mock_chroma_client.create_collection.assert_called_once()
            mock_chroma_client.get_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_file_change_detection(self, mock_embedding_model, mock_chroma_client):
        """Test file change detection (modified files)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test markdown file
            file_path = os.path.join(temp_dir, "test.md")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("# Test Header\n\nThis is test content.")
            
            db_path = os.path.join(temp_dir, "db")
            
            # First processing
            await process_files(
                temp_dir,
                "test_collection",
                mock_embedding_model,
                mock_chroma_client,
                db_path
            )
            
            # Reset mock to check calls in second run
            collection = MagicMock()
            mock_chroma_client.get_collection.return_value = collection
            mock_chroma_client.reset_mock()
            
            # Second processing without changes
            await process_files(
                temp_dir,
                "test_collection",
                mock_embedding_model,
                mock_chroma_client,
                db_path
            )
            
            # Collection.add should not be called since no files changed
            collection.add.assert_not_called()
            
            # Modify the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("# Test Header\n\nThis is updated content.")
            
            # Reset mock again
            collection.reset_mock()
            
            # Third processing with changes
            await process_files(
                temp_dir,
                "test_collection",
                mock_embedding_model,
                mock_chroma_client,
                db_path
            )
            
            # Collection.add should be called since file was modified
            collection.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_path(self, mock_embedding_model, mock_chroma_client):
        """Test error handling for invalid paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "db")
            
            # Call with non-existent path
            invalid_path = os.path.join(temp_dir, "nonexistent")
            result = await process_files(
                invalid_path,
                "test_collection",
                mock_embedding_model,
                mock_chroma_client,
                db_path
            )
            
            # Should return False for invalid path
            assert result is False

    @pytest.mark.asyncio
    async def test_large_file_handling(self, mock_embedding_model, mock_chroma_client):
        """Test handling of large files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a large markdown file (100KB)
            file_path = os.path.join(temp_dir, "large.md")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("# Large File\n\n")
                # Generate 100KB of content with headers and paragraphs
                for i in range(100):
                    f.write(f"## Section {i}\n\n")
                    f.write(f"This is paragraph {i} with some content. " * 20)
                    f.write("\n\n")
            
            db_path = os.path.join(temp_dir, "db")
            
            # Mock the embedding model to return appropriate shape for many chunks
            # Create a mock that can handle variable input sizes
            def mock_encode(texts):
                return np.random.rand(len(texts), 3)
            
            mock_embedding_model.encode.side_effect = mock_encode
            
            # Call process_files
            result = await process_files(
                file_path,
                "test_collection",
                mock_embedding_model,
                mock_chroma_client,
                db_path
            )
            
            # Verify results
            assert result is True
            collection = mock_chroma_client.create_collection.return_value
            # Should have called add at least once
            collection.add.assert_called_once()
            # The large file should be split into multiple chunks
            args, kwargs = collection.add.call_args
            assert len(kwargs.get('documents', [])) > 1
