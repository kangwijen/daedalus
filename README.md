# Daedalus

**Daedalus** is an intelligent assistant for designing Capture The Flag (CTF) challenges across multiple domains. Powered by Anthropic's Claude 3.7 and Voyage AI, it assists problem setters in creating high-quality CTF challenges.

## Features

- Expert CTF Challenge Generation using Claude 3.7
- Context-Aware Document Indexing with Change Detection
- Semantic Search + Reranking using Voyage AI
- Conversational Query Interface using Gradio
- Persistent Embedding Storage using ChromaDB
- Contextual Chat Memory to Keep Track of Previous Interactions

## Setup

1. Clone the repository
2. Install the dependencies
3. Set up the environment variables
4. Input context files (text or markdown files) to the `input_file` folder
5. Run the application with `python main.py`

## Usage

Once the application is running, you can start asking questions about CTF challenge design.

## Folder Structure

```
├── input_file/                     # Folder containing text/markdown files
    ├── example_subfolder/          # Subfolder containing text/markdown files
        └── example_file.md         # Example file in subfolder
    └── example_file.txt            # Example file in root folder
├── chroma_db/                      # Vector database (auto-created)
├── main.py                         # Main application
├── .env                            # API keys and config
└── README.md                       # Project overview
```
