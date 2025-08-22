import os
import asyncio
import zipfile
import re
from io import BytesIO
from dataclasses import dataclass
from typing import List, Dict, Callable

import streamlit as st
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import (
    BFSDeepCrawlStrategy,
    DFSDeepCrawlStrategy,
    BestFirstCrawlingStrategy,
)
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.documents import Document


# ---------------------------- Streamlit Page Config ----------------------------
st.set_page_config(
    page_title="Go-Fetch - Web Crawler & Chat",
    page_icon=":mag_right:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------ Data & Strategy -------------------------------
@dataclass
class CrawlerSettings:
    url: str
    max_depth: int
    max_pages: int
    strategy: str
    include_external: bool
    keywords: List[str]
    verbose: bool


def extract_title(markdown: str, url: str) -> str:
    match = re.search(r"^# (.+)", markdown, re.MULTILINE)
    title = match[1].strip() if match else url.rstrip("/").split("/")[-1] or "Untitled_Page"
    return re.sub(r"[^a-zA-Z0-9_-]", "_", title)


def build_strategy(settings: CrawlerSettings):
    if settings.strategy == "BFS":
        return BFSDeepCrawlStrategy(
            max_depth=settings.max_depth,
            include_external=settings.include_external,
            max_pages=settings.max_pages,
        )
    elif settings.strategy == "BestFirst":
        scorer = KeywordRelevanceScorer(keywords=settings.keywords, weight=0.7) if settings.keywords else None
        return BestFirstCrawlingStrategy(
            max_depth=settings.max_depth,
            include_external=settings.include_external,
            max_pages=settings.max_pages,
            url_scorer=scorer,
        )
    elif settings.strategy == "DFS":
        return DFSDeepCrawlStrategy(
            max_depth=settings.max_depth,
            include_external=settings.include_external,
            max_pages=settings.max_pages,
        )
    else:
        raise ValueError(f"Unknown strategy: {settings.strategy}")


# ---------------------------- Core Crawling Logic -----------------------------
def update_progress_ui(current: int, total: int, current_url: str):
    progress = min(current / total, 1.0)
    st.session_state.progress_bar.progress(progress)
    st.session_state.status_text.text(f"Crawled {current}/{total}: {current_url}")


async def crawl_website_async(
    settings: CrawlerSettings,
    progress_cb: Callable[[int, int, str], None],
) -> List[Dict]:
    config = CrawlerRunConfig(
        deep_crawl_strategy=build_strategy(settings),
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=settings.verbose,
        stream=True,
    )
    pages: List[Dict] = []
    async with AsyncWebCrawler() as crawler:
        async for result in await crawler.arun(settings.url, config=config):
            pages.append({"url": result.url, "markdown": result.markdown})
            progress_cb(len(pages), settings.max_pages, result.url)
            if len(pages) >= settings.max_pages:
                break
    return pages


def run_crawl(settings: CrawlerSettings):
    return asyncio.run(crawl_website_async(settings, update_progress_ui))


# -------------------------- Indexing & Retrieval (RAG) ------------------------
def build_index(
    pages: List[Dict],
    openai_api_key: str | None,
    embedding_model: str,
):
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs: List[Document] = []
    for p in pages:
        chunks = text_splitter.split_text(p["markdown"])
        docs.extend([Document(page_content=c, metadata={"url": p["url"]}) for c in chunks])

    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=openai_api_key if openai_api_key else None)
    vectorstore = Chroma.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return vectorstore, retriever


def create_qa_chain(retriever, llm_model: str, openai_api_key: str | None):
    llm = ChatOpenAI(model=llm_model, temperature=0, api_key=openai_api_key if openai_api_key else None)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return qa


# ------------------------------- UI Components --------------------------------
def render_sidebar() -> CrawlerSettings:
    st.sidebar.header("Crawl Settings")
    url = st.sidebar.text_input("Website URL", value="https://python.langchain.com/docs/")
    max_depth = st.sidebar.slider("Max Crawl Depth", 1, 10, 1)
    max_pages = st.sidebar.number_input("Max Pages to Crawl", 5, 1000, 10)
    strategy = st.sidebar.selectbox("Strategy", ["BFS", "DFS", "BestFirst"])
    include_external = st.sidebar.checkbox("Include External Links")
    keywords_input = st.sidebar.text_input("Keywords (comma‑sep)", "")
    verbose = st.sidebar.checkbox("Verbose Mode", value=True)

    st.sidebar.header("OpenAI & RAG")
    openai_api_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password")
    embed_model = st.sidebar.selectbox("Embedding Model", ["text-embedding-3-small", "text-embedding-3-large"], index=0)
    llm_model = st.sidebar.selectbox("LLM Model", ["gpt-4o-mini", "gpt-4o"], index=0)
    auto_index = st.sidebar.checkbox("Auto-build index after crawl", value=True)

    st.session_state.openai_api_key = openai_api_key
    st.session_state.embed_model = embed_model
    st.session_state.llm_model = llm_model
    st.session_state.auto_index = auto_index

    return CrawlerSettings(
        url=url,
        max_depth=max_depth,
        max_pages=max_pages,
        strategy=strategy,
        include_external=include_external,
        keywords=[k.strip() for k in keywords_input.split(",") if k.strip()],
        verbose=verbose,
    )


def render_download_buttons():
    combined_md = "\n\n---\n\n".join(f"# {p['url']}\n\n{p['markdown']}" for p in st.session_state.pages)
    st.download_button("Download All as Markdown", combined_md, "site.md", "text/markdown")

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, page in enumerate(st.session_state.pages, 1):
            filename = f"{i:03d}_{extract_title(page['markdown'], page['url'])}.md"
            zf.writestr(filename, page["markdown"])
    st.download_button("Download ZIP of Pages", buf.getvalue(), "pages.zip", "application/zip")


def render_preview():
    st.subheader("Preview First Page")
    with st.expander("Expand Preview"):
        st.write(st.session_state.pages[0]["markdown"][:5000])


def render_chat():
    st.subheader("Chat with Crawled Content")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
    with col2:
        if st.button("Rebuild Index"):
            if st.session_state.pages:
                vs, retr = build_index(
                    st.session_state.pages,
                    st.session_state.openai_api_key,
                    st.session_state.embed_model,
                )
                st.session_state.vectorstore = vs
                st.session_state.retriever = retr
                st.session_state.qa_chain = create_qa_chain(
                    retr, st.session_state.llm_model, st.session_state.openai_api_key
                )
    with col3:
        if st.button("Reset RAG (drop index)"):
            st.session_state.vectorstore = None
            st.session_state.retriever = None
            st.session_state.qa_chain = None

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    if st.session_state.pages and st.session_state.retriever is None:
        vs, retr = build_index(
            st.session_state.pages,
            st.session_state.openai_api_key,
            st.session_state.embed_model,
        )
        st.session_state.vectorstore = vs
        st.session_state.retriever = retr
        st.session_state.qa_chain = create_qa_chain(
            retr, st.session_state.llm_model, st.session_state.openai_api_key
        )

    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_msg = st.chat_input("Ask a question about the crawled site...")
    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        st.chat_message("user").markdown(user_msg)

        if st.session_state.qa_chain is None:
            st.chat_message("assistant").markdown(
                "Index not built yet. Crawl a site and/or click Rebuild Index."
            )
            return

        result = st.session_state.qa_chain({"query": user_msg})
        answer = result["result"]
        srcs = result.get("source_documents", []) or []
        src_lines = []
        for i, d in enumerate(srcs, 1):
            url = d.metadata.get("url", "")
            preview = d.page_content[:120].replace("\n", " ")
            src_lines.append(f"{i}. {url} — {preview}...")

        full_answer = answer
        if src_lines:
            full_answer += "\n\nSources:\n" + "\n".join(f"- {s}" for s in src_lines)

        st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
        st.chat_message("assistant").markdown(full_answer)


# ------------------------------- Main App -------------------------------------
def main():
    st.title("Website Crawler + Chat (Crawl4AI + LangChain + Chroma)")
    st.markdown("Enter a URL, crawl pages, then chat with the extracted markdown.")

    settings = render_sidebar()

    if "pages" not in st.session_state:
        st.session_state.pages = []
    if "progress_bar" not in st.session_state:
        st.session_state.progress_bar = st.progress(0)
    if "status_text" not in st.session_state:
        st.session_state.status_text = st.empty()
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Start Crawling", type="primary"):
            pages = run_crawl(settings)
            st.success(f"Crawled {len(pages)} pages!")
            st.session_state.pages = pages

            if st.session_state.auto_index and pages:
                vs, retr = build_index(
                    pages,
                    st.session_state.openai_api_key,
                    st.session_state.embed_model,
                )
                st.session_state.vectorstore = vs
                st.session_state.retriever = retr
                st.session_state.qa_chain = create_qa_chain(
                    retr, st.session_state.llm_model, st.session_state.openai_api_key
                )

    if st.session_state.pages:
        with st.container():
            render_download_buttons()
            render_preview()

    st.markdown("---")
    render_chat()

    st.markdown("Note: set OPENAI_API_KEY env var or enter it in the sidebar. "
                "Install deps: pip install crawl4ai langchain langchain-openai langchain-community chromadb")


if __name__ == "__main__":
    main()
