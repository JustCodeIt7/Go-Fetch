import streamlit as st
import requests
from bs4 import BeautifulSoup
import markdownify
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import os
import tempfile
import uuid
from typing import List, Dict, Any
from urllib.parse import urljoin, urlparse

# Page configuration
st.set_page_config(
    page_title="Enhanced Web RAG Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better chat UI
st.markdown("""
<style>
.stChatMessage {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 10px;
    margin: 5px 0;
}

.user-message {
    background-color: #e3f2fd;
    margin-left: 20%;
}

.assistant-message {
    background-color: #f3e5f5;
    margin-right: 20%;
}

.chat-container {
    max-height: 500px;
    overflow-y: auto;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

class WebCrawler:
    """Enhanced web crawler with better error handling and content extraction."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def crawl_url(self, url: str, max_depth: int = 2) -> Dict[str, Any]:
        """Crawl a URL and extract content with improved error handling."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract text content
            text_content = soup.get_text()
            
            # Convert to markdown
            markdown_content = markdownify.markdownify(str(soup), heading_style="ATX")
            
            # Clean up the content
            lines = markdown_content.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.isspace():
                    cleaned_lines.append(line)
            
            cleaned_markdown = '\n'.join(cleaned_lines)
            
            return {
                'url': url,
                'title': soup.title.string if soup.title else 'No Title',
                'content': text_content,
                'markdown': cleaned_markdown,
                'word_count': len(text_content.split()),
                'status': 'success'
            }
            
        except requests.RequestException as e:
            return {
                'url': url,
                'error': str(e),
                'status': 'error'
            }
        except Exception as e:
            return {
                'url': url,
                'error': f"Unexpected error: {str(e)}",
                'status': 'error'
            }

class RAGChatSystem:
    """RAG Chat System using ChromaDB and LangChain."""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.EphemeralClient()
        self.collection_name = f"web_content_{uuid.uuid4().hex[:8]}"
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize memory for conversation
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        self.vectorstore = None
        self.qa_chain = None
    
    def create_vectorstore(self, documents: List[str], metadatas: List[Dict] = None):
        """Create ChromaDB vectorstore from documents."""
        try:
            # Split documents into chunks
            all_chunks = []
            all_metadatas = []
            
            for i, doc in enumerate(documents):
                chunks = self.text_splitter.split_text(doc)
                all_chunks.extend(chunks)
                
                # Create metadata for each chunk
                base_metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                chunk_metadatas = [
                    {**base_metadata, "chunk_id": j, "total_chunks": len(chunks)}
                    for j in range(len(chunks))
                ]
                all_metadatas.extend(chunk_metadatas)
            
            # Create Chroma vectorstore
            self.vectorstore = Chroma.from_texts(
                texts=all_chunks,
                embedding=self.embeddings,
                metadatas=all_metadatas,
                collection_name=self.collection_name
            )
            
            return True
            
        except Exception as e:
            st.error(f"Error creating vectorstore: {str(e)}")
            return False
    
    def setup_qa_chain(self):
        """Setup the conversational QA chain."""
        if not self.vectorstore:
            return False
        
        try:
            # Create retriever
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            # Create conversational QA chain
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=True
            )
            
            return True
            
        except Exception as e:
            st.error(f"Error setting up QA chain: {str(e)}")
            return False
    
    def chat(self, question: str) -> Dict[str, Any]:
        """Process a chat question and return response."""
        if not self.qa_chain:
            return {
                "answer": "Please crawl and process some web content first.",
                "source_documents": []
            }
        
        try:
            response = self.qa_chain({"question": question})
            return response
            
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "source_documents": []
            }

def main():
    st.title("🤖 Enhanced Web RAG Chat Application")
    st.markdown("Crawl websites, extract content, and chat with the information using AI!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OpenAI API Key
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to enable chat functionality"
        )
        
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key to use chat features.")
            return
        
        st.header("🌐 Web Crawling")
        url_input = st.text_input(
            "Enter URL to crawl:",
            placeholder="https://example.com"
        )
        
        max_depth = st.slider("Crawl Depth", 1, 3, 1)
        
        if st.button("🕷️ Crawl Website", type="primary"):
            if url_input:
                with st.spinner("Crawling website..."):
                    crawler = WebCrawler()
                    result = crawler.crawl_url(url_input, max_depth)
                    st.session_state.crawl_result = result
            else:
                st.error("Please enter a URL to crawl.")
    
    # Initialize session state
    if "crawl_result" not in st.session_state:
        st.session_state.crawl_result = None
    
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "vectorstore_ready" not in st.session_state:
        st.session_state.vectorstore_ready = False
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Crawled Content")
        
        if st.session_state.crawl_result:
            result = st.session_state.crawl_result
            
            if result['status'] == 'success':
                st.success(f"✅ Successfully crawled: {result['title']}")
                st.info(f"Word count: {result['word_count']}")
                
                # Show content preview
                with st.expander("📖 Content Preview"):
                    st.text_area(
                        "Raw content (first 1000 chars):",
                        result['content'][:1000] + "..." if len(result['content']) > 1000 else result['content'],
                        height=200
                    )
                
                with st.expander("📝 Markdown Preview"):
                    st.text_area(
                        "Markdown content (first 1000 chars):",
                        result['markdown'][:1000] + "..." if len(result['markdown']) > 1000 else result['markdown'],
                        height=200
                    )
                
                # Process content for RAG
                if st.button("🧠 Process for Chat", type="secondary"):
                    with st.spinner("Processing content for chat..."):
                        # Initialize RAG system
                        rag_system = RAGChatSystem(openai_api_key)
                        
                        # Create vectorstore
                        documents = [result['markdown']]
                        metadatas = [{'url': result['url'], 'title': result['title']}]
                        
                        if rag_system.create_vectorstore(documents, metadatas):
                            if rag_system.setup_qa_chain():
                                st.session_state.rag_system = rag_system
                                st.session_state.vectorstore_ready = True
                                st.success("✅ Content processed successfully! You can now chat.")
                            else:
                                st.error("Failed to setup QA chain.")
                        else:
                            st.error("Failed to create vectorstore.")
            
            else:
                st.error(f"❌ Failed to crawl: {result.get('error', 'Unknown error')}")
        
        else:
            st.info("👆 Enter a URL in the sidebar to start crawling.")
    
    with col2:
        st.header("💬 Chat with Content")
        
        if st.session_state.vectorstore_ready and st.session_state.rag_system:
            # Chat interface
            chat_container = st.container()
            
            with chat_container:
                # Display chat history
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        with st.chat_message("user"):
                            st.write(message["content"])
                    else:
                        with st.chat_message("assistant"):
                            st.write(message["content"])
                            if "sources" in message:
                                with st.expander("📚 Sources"):
                                    for i, source in enumerate(message["sources"]):
                                        st.write(f"**Source {i+1}:**")
                                        st.write(source[:200] + "..." if len(source) > 200 else source)
            
            # Chat input
            if prompt := st.chat_input("Ask a question about the crawled content..."):
                # Add user message to history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": prompt
                })
                
                # Display user message
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = st.session_state.rag_system.chat(prompt)
                        
                        answer = response.get("answer", "I couldn't generate a response.")
                        source_docs = response.get("source_documents", [])
                        
                        st.write(answer)
                        
                        # Show sources if available
                        if source_docs:
                            with st.expander("📚 Sources"):
                                for i, doc in enumerate(source_docs):
                                    st.write(f"**Source {i+1}:**")
                                    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                                    st.write(content[:200] + "..." if len(content) > 200 else content)
                        
                        # Add assistant message to history
                        assistant_message = {
                            "role": "assistant",
                            "content": answer
                        }
                        
                        if source_docs:
                            assistant_message["sources"] = [
                                doc.page_content if hasattr(doc, 'page_content') else str(doc)
                                for doc in source_docs
                            ]
                        
                        st.session_state.chat_history.append(assistant_message)
                
                # Rerun to update chat display
                st.rerun()
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                if st.session_state.rag_system:
                    st.session_state.rag_system.memory.clear()
                st.rerun()
        
        else:
            st.info("💡 Process some crawled content first to enable chat functionality.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit, LangChain, ChromaDB, and OpenAI"
    )

if __name__ == "__main__":
    main()
