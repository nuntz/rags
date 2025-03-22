"""Tests for utility functions in the RAGS system."""

import os
import json
import tempfile
import pytest
from pathlib import Path
import hashlib

from rags.utils import (
    calculate_file_hash,
    split_markdown_into_chunks,
    load_file_registry,
    save_file_registry
)

class TestFileHashCalculation:
    def test_empty_file(self):
        """Test hash calculation with an empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            pass
        
        try:
            file_hash = calculate_file_hash(temp_file.name)
            # Known MD5 hash for empty file
            assert file_hash == "d41d8cd98f00b204e9800998ecf8427e"
        finally:
            os.unlink(temp_file.name)
    
    def test_known_content(self):
        """Test hash calculation with known content."""
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
            temp_file.write("Hello, world!")
        
        try:
            file_hash = calculate_file_hash(temp_file.name)
            # MD5 hash for "Hello, world!"
            assert file_hash == "6cd3556deb0da54bca060b4c39479839"
        finally:
            os.unlink(temp_file.name)
    
    def test_nonexistent_file(self):
        """Test error handling for non-existent files."""
        with pytest.raises(FileNotFoundError):
            calculate_file_hash("/path/to/nonexistent/file")


class TestMarkdownChunking:
    def test_empty_text(self):
        """Test chunking with empty text."""
        chunks = split_markdown_into_chunks("", "test.md")
        assert len(chunks) == 0
    
    def test_simple_markdown_no_headers(self):
        """Test with simple markdown without headers."""
        text = "This is a simple paragraph.\n\nThis is another paragraph."
        chunks = split_markdown_into_chunks(text, "test.md")
        
        assert len(chunks) == 1
        content, metadata = chunks[0]
        assert "simple paragraph" in content
        assert "another paragraph" in content
        assert metadata["source"] == "test.md"
        assert metadata["header"] == ""
    
    def test_various_header_levels(self):
        """Test with various header levels (h1-h6)."""
        text = """# H1 Header
Content under H1.

## H2 Header
Content under H2.

### H3 Header
Content under H3.

#### H4 Header
Content under H4.

##### H5 Header
Content under H5.

###### H6 Header
Content under H6."""
        
        chunks = split_markdown_into_chunks(text, "test.md")
        
        # Should have 6 chunks, one for each header section
        assert len(chunks) == 6
        
        # Check first chunk
        content, metadata = chunks[0]
        assert "Content under H1" in content
        assert metadata["header"] == "# H1 Header"
        
        # Check last chunk
        content, metadata = chunks[5]
        assert "Content under H6" in content
        assert metadata["header"] == "###### H6 Header"
    
    def test_exceeding_max_chunk_size(self):
        """Test with content exceeding max_chunk_size."""
        # Create a long paragraph
        long_paragraph = "word " * 600  # ~3000 characters
        
        text = f"""# Header
{long_paragraph}

Second paragraph."""
        
        chunks = split_markdown_into_chunks(text, "test.md", max_chunk_size=1000, overlap=200)
        
        # Should be split into multiple chunks
        assert len(chunks) > 1
        
        # Check for overlap between chunks
        content1, _ = chunks[0]
        content2, _ = chunks[1]
        
        # The end of the first chunk should appear at the beginning of the second
        words1 = content1.split()
        words2 = content2.split()
        
        overlap_words = words1[-40:]  # Take more than overlap/5 words to ensure we catch the overlap
        for word in overlap_words:
            if word in words2[:40]:  # Check in the first 40 words of the second chunk
                break
        else:
            pytest.fail("No overlap found between chunks")
    
    def test_different_overlap_settings(self):
        """Test with different overlap settings."""
        # Create a paragraph with distinct words - using more words to ensure chunking
        # Each word is about 6 chars, so 200 words is ~1200 chars
        words = [f"word{i}" for i in range(200)]
        text = "# Header\n" + " ".join(words)
        
        # Test with small overlap - max_chunk_size small enough to force multiple chunks
        chunks_small = split_markdown_into_chunks(text, "test.md", max_chunk_size=400, overlap=50)
        
        # Test with large overlap
        chunks_large = split_markdown_into_chunks(text, "test.md", max_chunk_size=400, overlap=200)
        
        # Both should have multiple chunks
        assert len(chunks_small) > 1
        assert len(chunks_large) > 1
        
        # Large overlap should have more words in common between chunks
        content_small_1, _ = chunks_small[0]
        content_small_2, _ = chunks_small[1]
        content_large_1, _ = chunks_large[0]
        content_large_2, _ = chunks_large[1]
        
        common_words_small = set(content_small_1.split()) & set(content_small_2.split())
        common_words_large = set(content_large_1.split()) & set(content_large_2.split())
        
        assert len(common_words_large) > len(common_words_small)
    
    def test_multi_level_header_nesting(self):
        """Test with multi-level header nesting."""
        text = """# Main Header
Main content.

## Sub Header 1
Sub content 1.

### Sub-sub Header
Sub-sub content.

## Sub Header 2
Sub content 2."""
        
        chunks = split_markdown_into_chunks(text, "test.md")
        
        assert len(chunks) == 4
        
        # Check headers are correctly assigned
        _, metadata1 = chunks[0]
        _, metadata2 = chunks[1]
        _, metadata3 = chunks[2]
        _, metadata4 = chunks[3]
        
        assert metadata1["header"] == "# Main Header"
        assert metadata2["header"] == "## Sub Header 1"
        assert metadata3["header"] == "### Sub-sub Header"
        assert metadata4["header"] == "## Sub Header 2"
    
    def test_metadata_generation(self):
        """Test correct metadata generation for chunks."""
        text = """# Header
Content."""
        
        filename = Path("/path/to/test.md")
        chunks = split_markdown_into_chunks(text, filename)
        
        assert len(chunks) == 1
        _, metadata = chunks[0]
        
        assert metadata["source"] == str(filename)
        assert metadata["header"] == "# Header"


class TestFileRegistry:
    def test_load_nonexistent_registry(self):
        """Test loading a non-existent registry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir) / "nonexistent_registry.json"
            registry = load_file_registry(registry_path)
            assert registry == {}
    
    def test_save_load_roundtrip(self):
        """Test round-trip functionality: save then load registry."""
        registry_data = {
            "file1.md": {"hash": "abc123", "last_updated": "2023-01-01"},
            "file2.md": {"hash": "def456", "last_updated": "2023-01-02"}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir) / "registry.json"
            
            # Save the registry
            save_file_registry(registry_data, registry_path)
            
            # Load it back
            loaded_registry = load_file_registry(registry_path)
            
            # Verify it's the same
            assert loaded_registry == registry_data
    
    def test_complex_nested_registry(self):
        """Test with complex nested registry data."""
        registry_data = {
            "folder1": {
                "file1.md": {"hash": "abc123", "chunks": [1, 2, 3]},
                "file2.md": {"hash": "def456", "chunks": [4, 5, 6]}
            },
            "folder2": {
                "file3.md": {
                    "hash": "ghi789", 
                    "metadata": {
                        "author": "John Doe",
                        "date": "2023-01-03"
                    }
                }
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir) / "registry.json"
            
            # Save the registry
            save_file_registry(registry_data, registry_path)
            
            # Load it back
            loaded_registry = load_file_registry(registry_path)
            
            # Verify it's the same
            assert loaded_registry == registry_data
            
            # Verify nested structure is preserved
            assert loaded_registry["folder1"]["file1.md"]["chunks"] == [1, 2, 3]
            assert loaded_registry["folder2"]["file3.md"]["metadata"]["author"] == "John Doe"
    
    def test_permission_error(self, monkeypatch):
        """Test error handling for permission issues."""
        # Mock open to raise PermissionError
        def mock_open(*args, **kwargs):
            raise PermissionError("Permission denied")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir) / "registry.json"
            
            # Create an empty file
            registry_path.touch()
            
            # Mock the built-in open function
            monkeypatch.setattr("builtins.open", mock_open)
            
            # Test load_file_registry
            with pytest.raises(PermissionError):
                load_file_registry(registry_path)
            
            # Test save_file_registry
            with pytest.raises(PermissionError):
                save_file_registry({}, registry_path)
