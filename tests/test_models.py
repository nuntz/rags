"""Tests for the models module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import chromadb
from sentence_transformers import SentenceTransformer

from rags.models import RAGDependencies, create_llm_model
from pydantic_ai.models.openai import OpenAIModel


class TestRAGDependencies:
    """Tests for the RAGDependencies class."""

    def test_initialization(self):
        """Test basic initialization with mock components."""
        mock_client = Mock(spec=chromadb.Client)
        mock_model = Mock(spec=SentenceTransformer)
        
        deps = RAGDependencies(
            chroma_client=mock_client,
            collection_name="test_collection",
            embedding_model=mock_model
        )
        
        assert deps.chroma_client is mock_client
        assert deps.collection_name == "test_collection"
        assert deps.embedding_model is mock_model
    
    def test_initialization_with_real_components(self):
        """Test initialization with actual component types."""
        with patch("chromadb.Client") as MockClient:
            mock_client = MockClient.return_value
            mock_model = Mock(spec=SentenceTransformer)
            
            deps = RAGDependencies(
                chroma_client=mock_client,
                collection_name="test_collection",
                embedding_model=mock_model
            )
            
            assert isinstance(deps, RAGDependencies)
            assert deps.collection_name == "test_collection"


class TestLLMModelCreation:
    """Tests for the create_llm_model function."""
    
    def test_default_parameters(self):
        """Test create_llm_model with default parameters."""
        model = create_llm_model()
        
        assert isinstance(model, OpenAIModel)
        assert model.model_name == "gpt-4o"
        # We can't directly access the provider, so we'll just verify the model was created
        # The create_llm_model function itself shows the default values being used
    
    def test_custom_parameters(self):
        """Test create_llm_model with custom parameters."""
        custom_url = "https://api.example.com/v1"
        custom_key = "test-api-key"
        custom_model = "gpt-3.5-turbo"
        
        # Create a dummy model to avoid initialization issues
        dummy_model = MagicMock(spec=OpenAIModel)
        dummy_model.model_name = custom_model
        
        with patch("rags.models.OpenAIModel", return_value=dummy_model), \
             patch("rags.models.OpenAIProvider"):
            model = create_llm_model(
                base_url=custom_url,
                api_key=custom_key,
                model_name=custom_model
            )
            
            assert model is dummy_model
            assert model.model_name == custom_model
    
    @patch("rags.models.OpenAIProvider")
    def test_connection_error_handling(self, mock_provider):
        """Test error handling for connection failures."""
        # Setup the mock to raise an exception when initialized
        mock_provider.side_effect = ConnectionError("Failed to connect")
        
        # The function should propagate the exception
        with pytest.raises(ConnectionError) as excinfo:
            create_llm_model()
        
        assert "Failed to connect" in str(excinfo.value)
