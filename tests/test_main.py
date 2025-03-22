"""Tests for the main entry point and CLI functionality."""

import os
import sys
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

from rags.main import main, main_cli
from rags.version import __version__

class TestCommandLineArguments:
    """Tests for command line argument handling."""
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('builtins.print')
    @patch('rags.main.create_llm_model')
    @patch('rags.main.process_files')
    @patch('chromadb.PersistentClient')
    @patch('sentence_transformers.SentenceTransformer')
    async def test_version_flag(self, mock_transformer, mock_chroma, mock_process, 
                               mock_llm, mock_print, mock_args):
        """Test that --version flag displays version and exits."""
        # Setup
        mock_args.return_value.version = True
        mock_args.return_value.docs_path = None
        
        # Execute
        await main()
        
        # Assert
        mock_print.assert_called_with(f"RAGS version {__version__}")
        mock_process.assert_not_called()
        mock_llm.assert_not_called()
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('argparse.ArgumentParser.print_help')
    @patch('builtins.print')
    @patch('rags.main.process_files')
    async def test_missing_docs_path(self, mock_process, mock_print, 
                                    mock_print_help, mock_args):
        """Test behavior when required docs_path is missing."""
        # Setup
        mock_args.return_value.version = False
        mock_args.return_value.docs_path = None
        
        # Execute
        await main()
        
        # Assert
        mock_print_help.assert_called_once()
        mock_print.assert_called_with("\nError: docs_path is required unless --version is specified")
        mock_process.assert_not_called()
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('os.makedirs')
    @patch('rags.main.process_files')
    @patch('chromadb.PersistentClient')
    @patch('sentence_transformers.SentenceTransformer')
    async def test_custom_arguments(self, mock_transformer, mock_chroma, 
                                   mock_process, mock_makedirs, mock_args):
        """Test that custom arguments are properly passed to components."""
        # Setup
        mock_args.return_value.version = False
        mock_args.return_value.docs_path = "/test/docs"
        mock_args.return_value.force_reload = True
        mock_args.return_value.collection_name = "test_collection"
        mock_args.return_value.db_path = "/test/db"
        mock_args.return_value.request_limit = 100
        mock_args.return_value.llm_url = "http://test-llm:8080/v1"
        mock_args.return_value.api_key = "test-key"
        mock_args.return_value.model_name = "test-model"
        
        # Mock process_files to return False to exit early
        mock_process.return_value = False
        
        # Execute
        await main()
        
        # Assert
        mock_makedirs.assert_called_with("/test/db", exist_ok=True)
        mock_chroma.assert_called_with(path="/test/db")
        mock_process.assert_called_with(
            "/test/docs", 
            "test_collection", 
            mock_transformer.return_value, 
            mock_chroma.return_value, 
            "/test/db",
            True
        )
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('rags.main.process_files')
    @patch('builtins.print')
    @patch('chromadb.PersistentClient')
    @patch('sentence_transformers.SentenceTransformer')
    async def test_process_files_failure(self, mock_transformer, mock_chroma, 
                                        mock_print, mock_process, mock_args):
        """Test handling of process_files failure."""
        # Setup
        mock_args.return_value.version = False
        mock_args.return_value.docs_path = "/test/docs"
        mock_args.return_value.force_reload = False
        mock_args.return_value.collection_name = "default_collection"
        mock_args.return_value.db_path = "~/.local/share/rags/chroma_db"
        
        # Mock process_files to return False (failure)
        mock_process.return_value = False
        
        # Execute
        await main()
        
        # Assert
        mock_print.assert_any_call("Error processing documentation files.")


class TestMainLoop:
    """Tests for the main interactive loop."""
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('rags.main.process_files')
    @patch('rags.main.create_llm_model')
    @patch('pydantic_ai.Agent')
    @patch('builtins.input')
    @patch('builtins.print')
    @patch('chromadb.PersistentClient')
    @patch('sentence_transformers.SentenceTransformer')
    async def test_exit_commands(self, mock_transformer, mock_chroma, mock_print, 
                               mock_input, mock_agent, mock_llm, mock_process, mock_args):
        """Test that exit and quit commands terminate the loop."""
        # Setup
        mock_args.return_value.version = False
        mock_args.return_value.docs_path = "/test/docs"
        mock_args.return_value.force_reload = False
        mock_args.return_value.collection_name = "default_collection"
        mock_args.return_value.db_path = "~/.local/share/rags/chroma_db"
        mock_args.return_value.request_limit = 50
        
        mock_process.return_value = True
        
        # Test with 'exit' command
        mock_input.return_value = "exit"
        
        # Execute
        await main()
        
        # Assert
        mock_agent.return_value.run.assert_not_called()
        
        # Reset mocks
        mock_input.reset_mock()
        mock_agent.reset_mock()
        
        # Test with 'quit' command
        mock_input.return_value = "quit"
        
        # Execute
        await main()
        
        # Assert
        mock_agent.return_value.run.assert_not_called()
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('rags.main.process_files')
    @patch('rags.main.create_llm_model')
    @patch('pydantic_ai.Agent')
    @patch('builtins.input')
    @patch('builtins.print')
    @patch('chromadb.PersistentClient')
    @patch('sentence_transformers.SentenceTransformer')
    async def test_query_processing(self, mock_transformer, mock_chroma, mock_print, 
                                  mock_input, mock_agent, mock_llm, mock_process, mock_args):
        """Test that user queries are processed correctly."""
        # Setup
        mock_args.return_value.version = False
        mock_args.return_value.docs_path = "/test/docs"
        mock_args.return_value.force_reload = False
        mock_args.return_value.collection_name = "default_collection"
        mock_args.return_value.db_path = "~/.local/share/rags/chroma_db"
        mock_args.return_value.request_limit = 50
        
        mock_process.return_value = True
        
        # Mock user input sequence: one query, then exit
        mock_input.side_effect = ["How does this work?", "exit"]
        
        # Mock agent response
        mock_agent_instance = MagicMock()
        mock_agent.return_value = mock_agent_instance
        
        mock_result = MagicMock()
        mock_result.data = "This is how it works..."
        mock_agent_instance.run = AsyncMock(return_value=mock_result)
        
        # Execute
        await main()
        
        # Assert
        mock_agent_instance.run.assert_called_once()
        call_args = mock_agent_instance.run.call_args[0]
        assert call_args[0] == "How does this work?"
        
        # Check that the result was printed
        mock_print.assert_any_call("This is how it works...")

    @patch('argparse.ArgumentParser.parse_args')
    @patch('rags.main.process_files')
    @patch('rags.main.create_llm_model')
    @patch('pydantic_ai.Agent')
    @patch('builtins.input')
    @patch('builtins.print')
    @patch('chromadb.PersistentClient')
    @patch('sentence_transformers.SentenceTransformer')
    async def test_exception_handling(self, mock_transformer, mock_chroma, mock_print, 
                                    mock_input, mock_agent, mock_llm, mock_process, mock_args):
        """Test that exceptions are properly caught and reported."""
        # Setup
        mock_args.return_value.version = False
        mock_args.return_value.docs_path = "/test/docs"
        
        # Simulate an exception during processing
        mock_process.side_effect = Exception("Test error")
        
        # Execute
        await main()
        
        # Assert
        mock_print.assert_any_call("An error occurred: Exception: Test error")


class TestMainCLI:
    """Tests for the main_cli function."""
    
    @patch('asyncio.run')
    def test_main_cli(self, mock_run):
        """Test that main_cli calls asyncio.run with main function."""
        # Execute
        main_cli()
        
        # Assert
        mock_run.assert_called_once()
        # Check that the first argument to run is the main coroutine
        assert mock_run.call_args[0][0].__name__ == 'main'
