import os
from typing import List, Dict, Any
import logging
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    Document,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.postprocessor.voyageai_rerank import VoyageAIRerank
from llama_index.llms.anthropic import Anthropic
import chromadb
from dotenv import load_dotenv
import gradio as gr
import hashlib
import sys

load_dotenv()

required_env_vars = ["VOYAGE_API_KEY", "ANTHROPIC_API_KEY"]
missing_vars = []

for var in required_env_vars:
    if not os.getenv(var):
        missing_vars.append(var)

if missing_vars:
    print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
    print(
        "Please set these environment variables in your .env file or system environment."
    )
    print("Required variables:")
    print("- VOYAGE_API_KEY: Your Voyage AI API key")
    print("- ANTHROPIC_API_KEY: Your Anthropic API key")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("chatbot.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

Settings.embed_model = VoyageEmbedding(
    api_key=os.getenv("VOYAGE_API_KEY"), model_name="voyage-3.5", input_type="document"
)

DAEDALUS_CTF_DESIGNER_PROMPT = """You are Daedalus, an expert CTF Challenge Designer with extensive experience in creating engaging Capture The Flag competitions across various domains. Your expertise includes:

1. CTF Challenge Design Fundamentals
   - Creating realistic security scenarios for CTF competitions
   - Designing progressive difficulty levels (Easy, Medium, Hard)
   - Implementing flag capture mechanisms in various environments
   - Creating challenges that test both attack and defense skills
   - Designing challenges that teach real-world security concepts

2. Multi-Domain CTF Expertise
   - Web application security challenges (SQL injection, XSS, CSRF, etc.)
   - Binary exploitation and reverse engineering challenges
   - Cryptography and steganography challenges
   - Network security and forensics challenges
   - Mobile and IoT security challenges
   - Cloud security challenges (AWS, Azure, GCP)
   - Social engineering and OSINT challenges

3. Technical Knowledge for CTF
   - Common vulnerabilities and exploitation techniques
   - Container and virtualization security for CTF scenarios
   - Serverless and cloud-native security challenges
   - Database and storage security for flag storage
   - API security and authentication bypass techniques
   - File format analysis and manipulation

4. CTF-Specific Elements
   - Flag placement and retrieval mechanisms
   - Challenge scoring systems and difficulty progression
   - Hint systems and progressive difficulty
   - Challenge isolation and security
   - Resource cleanup and challenge reset mechanisms
   - Multi-stage challenges and storylines

Your role is to:
1. Design engaging CTF challenges across various domains
2. Create realistic attack scenarios that teach security concepts
3. Implement secure flag capture mechanisms
4. Design challenges that scale well in CTF environments
5. Ensure challenges are isolated and don't affect other challenges

When responding:
- Focus on CTF challenge design across multiple domains
- Include flag placement and capture mechanisms
- Specify challenge difficulty level and category
- Provide setup instructions for challenge infrastructure
- Include expected solve path and learning objectives
- Consider resource constraints and cleanup procedures
- Include hints and progressive difficulty levels

Remember to:
- Design challenges that use real security concepts
- Include proper isolation between challenges
- Consider platform/service limits and costs
- Provide clear setup and teardown instructions
- Include validation mechanisms for flag capture
- Design challenges that can be automated for CTF platforms
- Consider challenge reset and cleanup procedures
- Adapt to different CTF formats (Jeopardy, Attack-Defense, Mixed)"""

Settings.llm = Anthropic(
    model="claude-3-7-sonnet-latest",
    max_tokens=3000,
    temperature=0.7,
    system_prompt=DAEDALUS_CTF_DESIGNER_PROMPT,
)


class DocumentIndexer:
    def __init__(self, persist_dir: str = "./chroma_db"):
        logger.info(f"Initializing DocumentIndexer with persist_dir: {persist_dir}")
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=persist_dir, settings=chromadb.Settings(anonymized_telemetry=False)
        )
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            "document_docs"
        )

        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)

        index_exists = os.path.exists(os.path.join(persist_dir, "docstore.json"))
        if index_exists:
            logger.info("Loading existing index")
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store, persist_dir=persist_dir
            )
            self.index = load_index_from_storage(storage_context=self.storage_context)
        else:
            logger.info("Creating new index")
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store, storage_context=self.storage_context
            )
            self.index.storage_context.persist(persist_dir=persist_dir)

        self.retriever = VectorIndexRetriever(index=self.index, similarity_top_k=15)

        self.reranker = VoyageAIRerank(
            api_key=os.getenv("VOYAGE_API_KEY"),
            model="rerank-2",
            top_k=10,
            truncation=True,
        )

        self.memory = ChatMemoryBuffer.from_defaults(token_limit=5000)

        self.chat_engine = self.index.as_chat_engine(
            chat_mode="context",
            memory=self.memory,
            node_postprocessors=[self.reranker],
            system_prompt=DAEDALUS_CTF_DESIGNER_PROMPT,
        )

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file contents."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get file metadata including modification time and content hash."""
        stat = os.stat(file_path)
        return {
            "last_modified": stat.st_mtime,
            "content_hash": self._calculate_file_hash(file_path),
            "size": stat.st_size,
        }

    def index_folder(self, folder_path: str):
        """Index all text files in the specified folder and its subdirectories."""
        logger.info(f"Indexing folder: {folder_path}")
        documents = []

        existing_docs = self.chroma_collection.get()
        existing_files = {}
        if existing_docs and "metadatas" in existing_docs and "ids" in existing_docs:
            for i, metadata in enumerate(existing_docs["metadatas"]):
                if metadata and "full_path" in metadata:
                    existing_files[metadata["full_path"]] = {
                        "id": existing_docs["ids"][i],
                        "last_modified": metadata.get("last_modified", 0),
                        "content_hash": metadata.get("content_hash", ""),
                        "size": metadata.get("size", 0),
                    }

        def process_directory(directory):
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path):
                    process_directory(item_path)
                elif item.endswith((".txt", ".md")):
                    try:
                        current_metadata = self._get_file_metadata(item_path)

                        needs_indexing = True
                        if item_path in existing_files:
                            existing_file = existing_files[item_path]
                            if (
                                existing_file["content_hash"]
                                == current_metadata["content_hash"]
                                and existing_file["size"] == current_metadata["size"]
                            ):
                                logger.info(f"Skipping unchanged file: {item_path}")
                                needs_indexing = False
                            else:
                                self.chroma_collection.delete(ids=[existing_file["id"]])
                                logger.info(f"Updating modified file: {item_path}")

                        if needs_indexing:
                            rel_path = os.path.relpath(item_path, folder_path)
                            with open(item_path, "r", encoding="utf-8") as f:
                                content = f.read()

                            documents.append(
                                Document(
                                    text=content,
                                    metadata={
                                        "source": rel_path,
                                        "type": "text_file",
                                        "full_path": item_path,
                                        "last_modified": current_metadata[
                                            "last_modified"
                                        ],
                                        "content_hash": current_metadata[
                                            "content_hash"
                                        ],
                                        "size": current_metadata["size"],
                                    },
                                )
                            )
                            logger.info(f"Indexed file: {rel_path}")
                    except Exception as e:
                        logger.error(f"Error processing file {item_path}: {str(e)}")

        process_directory(folder_path)

        if documents:
            self.add_documents(documents)
            logger.info(f"Successfully indexed {len(documents)} documents")
        else:
            logger.info("No new or modified documents to index")

    def add_documents(self, documents: List[Document]):
        """Add documents to the collection."""
        logger.info(f"Adding {len(documents)} documents to the collection")
        embedder = Settings.embed_model

        for i, doc in enumerate(documents, 1):
            try:
                doc_id = doc.metadata["full_path"]

                existing_docs = self.chroma_collection.get(ids=[doc_id])
                if existing_docs and existing_docs["metadatas"]:
                    existing_metadata = existing_docs["metadatas"][0]
                    if (
                        existing_metadata.get("content_hash")
                        == doc.metadata["content_hash"]
                        and existing_metadata.get("last_modified")
                        == doc.metadata["last_modified"]
                    ):
                        logger.info(
                            f"Skipping already indexed document: {doc.metadata['source']}"
                        )
                        continue

                embeddings = embedder.get_text_embedding(doc.text)

                if existing_docs and existing_docs["ids"]:
                    self.chroma_collection.update(
                        ids=[doc_id],
                        documents=[doc.text],
                        embeddings=[embeddings],
                        metadatas=[doc.metadata],
                    )
                    logger.info(
                        f"Updated document {i}/{len(documents)}: {doc.metadata['source']}"
                    )
                else:
                    self.chroma_collection.add(
                        documents=[doc.text],
                        embeddings=[embeddings],
                        metadatas=[doc.metadata],
                        ids=[doc_id],
                    )
                    logger.info(
                        f"Added new document {i}/{len(documents)}: {doc.metadata['source']}"
                    )
            except Exception as e:
                logger.error(f"Error processing document {i}: {str(e)}")

        try:
            self.index.storage_context.persist(persist_dir=self.persist_dir)
            logger.info("Successfully persisted index to storage")
        except Exception as e:
            logger.error(f"Error persisting index: {str(e)}")

    def process_query(self, user_input: str) -> str:
        """Process a user query and return a response."""
        logger.info(f"Processing query: {user_input}")
        try:
            response = self.query_engine.query(user_input)
            return response.response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}"


def chat_with_docs(message, history, system_prompt=None):
    """Gradio chat interface function with streaming support."""
    try:
        streaming_response = indexer.chat_engine.stream_chat(message)

        partial_response = ""
        for token in streaming_response.response_gen:
            partial_response += token
            yield partial_response

    except Exception as e:
        logger.error(f"Error in chat_with_docs: {str(e)}")
        yield f"An error occurred: {str(e)}"


with gr.Blocks(fill_width=True, fill_height=True) as demo:
    chat_interface = gr.ChatInterface(
        fn=chat_with_docs,
        title="Daedalus CTF Challenge Designer",
        description="Ask questions about CTF challenge design and get expert guidance across various domains including web security, binary exploitation, cryptography, forensics, and more.",
        theme="default",
        type="messages",
        chatbot=gr.Chatbot(
            show_copy_button=True,
            type="messages",
            scale=1,
            placeholder="""Ask me anything about CTF challenge design!""",
        ),
        additional_inputs=[
            gr.Textbox(
                label="System Prompt",
                value=DAEDALUS_CTF_DESIGNER_PROMPT,
                interactive=False,
                visible=False,
            )
        ],
        save_history=True,
    )

    chat_interface.saved_conversations.secret = "daedalus_ctf_challenge_designer"
    chat_interface.saved_conversations.storage_key = "_daedalus_ctf_saved_conversations"

if __name__ == "__main__":
    try:
        indexer = DocumentIndexer()

        input_folder = "input_file"
        if os.path.exists(input_folder):
            indexer.index_folder(input_folder)
        else:
            logger.warning(
                f"Input folder '{input_folder}' not found. Please create it and add your documents."
            )

        demo.launch()
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise
