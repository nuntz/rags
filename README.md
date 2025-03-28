# RAGS - Retrieval-Augmented Generation System

A Retrieval-Augmented Generation (RAG) system built with PydanticAI and Chroma DB that allows you to query your Markdown documentation using a local LLM server.

[![Python Tests](https://github.com/nuntz/rags/actions/workflows/python-tests.yml/badge.svg)](https://github.com/nuntz/rags/actions/workflows/python-tests.yml)

## Features

- **Document Processing**: Automatically processes Markdown files, breaking them into semantically meaningful chunks
- **Intelligent Caching**: Only processes new or changed files to save time
- **Vector Search**: Stores document embeddings in Chroma DB for semantic search
- **LLM Integration**: Connects to a local LLM server compatible with the OpenAI API
- **Interactive Interface**: Simple command-line interface for querying your documentation
- **XDG Compliant**: Follows the XDG Base Directory Specification for data storage

## Requirements

- Python 3.10+
- A local LLM server running with the `--jinja` flag (llama.cpp or similar)
- Markdown documentation files

## Installation

### From Source

1. Clone this repository:

```bash
git clone https://github.com/nuntz/rags
cd rags
```

2. Install the package:

```bash
pip install -e .
```

This will install the `rags` command-line tool and all required dependencies.

### Using pip (once published)

```bash
pip install rags
```

## Usage

### Basic Usage

```bash
rags /path/to/your/markdown/docs
```

### Options

- `--version`: Display the version number and exit
- `--force-reload`: Force reprocessing of all files, even if they haven't changed
- `--collection-name NAME`: Use a specific collection name in the vector database (default: "default_collection")
- `--db-path PATH`: Specify the path to store the ChromaDB database (default: "~/.local/share/rags/chroma_db")
- `--request-limit N`: Set the maximum number of requests to make to the LLM (default: 50)
- `--llm-url URL`: Specify the URL for the LLM server (default: "http://127.0.0.1:8080/v1")
- `--api-key KEY`: Provide an API key for the LLM server (default: "not-needed")
- `--model-name NAME`: Specify the model name to use (default: "gpt-4o")

### Examples

Process a directory of markdown files:
```bash
rags ./documentation
```

Process a specific markdown file:
```bash
rags ./documentation/getting-started.md
```

Force reprocessing of all files:
```bash
rags ./documentation --force-reload
```

Use a specific collection name and database location:
```bash
rags ./documentation --collection-name my_project_docs --db-path ~/myproject/vector_db
```

Set a custom request limit for the LLM:
```bash
rags ./documentation --request-limit 100
```

Connect to a different LLM server:
```bash
rags ./documentation --llm-url http://localhost:1234/v1 --api-key your-api-key --model-name llama3
```

## Data Storage

By default, the application stores data in the following locations:

- **Vector Database**: `~/.local/share/rags/chroma_db`
- **File Registry**: `~/.local/share/rags/data/file_registry.json`

The file registry tracks which files have been processed and their hash values to avoid unnecessary reprocessing.

## How It Works

1. **Document Processing**:
   - The script scans the provided directory for Markdown files
   - Each file is broken into smaller chunks based on headers and paragraphs
   - Files are tracked using MD5 hashes to detect changes

2. **Vector Database**:
   - Document chunks are converted to embeddings using SentenceTransformer
   - Embeddings and original text are stored in Chroma DB
   - File processing status is tracked in the file registry

3. **Query Processing**:
   - User questions are converted to embeddings
   - Similar document chunks are retrieved from Chroma DB
   - The LLM uses these relevant chunks to generate an answer

## LLM Server Requirements

The script is designed to work with local LLM servers that implement the OpenAI API, such as:

- llama.cpp (with OpenAI API compatibility)
- LM Studio
- Ollama (with OpenAI compatibility mode)
- LocalAI
- vLLM

**Important**: When using llama.cpp, make sure to start the server with the `--jinja` flag to enable function calling:

```bash
./llama-server -m /path/to/your/model.gguf --jinja --host 127.0.0.1 --port 8080
```

## Customization

The script can be easily modified to:
- Use different embedding models
- Adjust chunking parameters
- Connect to different LLM servers
- Define custom prompts

Look for the relevant parameters and functions in the code to make adjustments.

## Utility Functions

RAGS includes several utility functions that handle core functionality:

### File Hash Calculation
- Calculates MD5 hashes of files to detect changes
- Used to determine which files need reprocessing
- Handles various file types including empty and binary files

### Markdown Chunking
- Splits text into simple chunks of configurable size
- Implements overlap between chunks for better retrieval
- Handles documents of various sizes efficiently
- Uses a straightforward word-based chunking approach

### File Registry
- Tracks processed files and their metadata
- Stores file hashes and processing timestamps
- Supports nested directory structures

All utility functions are thoroughly tested with comprehensive test coverage.

## Limitations

- Designed for text content in Markdown format
- Performance depends on the quality of your local LLM and embedding model
- No support for authentication or HTTPS for the LLM server connection

## Development

### Testing

Run the test suite with pytest:

```bash
pytest
```

To run tests with coverage reporting:

```bash
pytest --cov=rags
```

## License

[MIT License](LICENSE)
