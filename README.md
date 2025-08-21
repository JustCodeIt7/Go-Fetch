# Go-Fetch - Web Crawler

Go-Fetch is a user-friendly web crawling application built with Streamlit and powered by the `crawl4ai` library. It provides an intuitive graphical interface to configure and execute web crawls, allowing you to extract content from websites and download it in various formats.

## Features

- **Configurable Crawl Settings**: Easily set the target URL, maximum crawl depth, and the number of pages to crawl.
- **Multiple Crawling Strategies**: Choose between Breadth-First Search (BFS), Depth-First Search (DFS), and Best-First crawling strategies to suit your needs.
- **External Link Inclusion**: Option to include or exclude external links during the crawl.
- **Keyword Relevance Scoring**: For the Best-First strategy, prioritize pages based on specified keywords.
- **Real-time Progress Updates**: Monitor the crawling process with live progress indicators.
- **Content Download**: Download all crawled content as a single Markdown file or as a ZIP archive containing individual Markdown files for each page.
- **Page Preview**: Quickly preview the content of the first crawled page directly within the application.

## Installation

To get started with Go-Fetch, follow these steps:

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/Go-Fetch.git
    cd Go-Fetch
    ```

    *(Note: Replace `https://github.com/your-username/Go-Fetch.git` with the actual repository URL if it's hosted elsewhere.)*

2.  **Install dependencies**:

    Go-Fetch requires `streamlit` and `crawl4ai`. You can install them using pip:

    ```bash
    pip install streamlit crawl4ai
    ```

3.  **Set up `crawl4ai`**:

    Before your first use, you need to run the `crawl4ai-setup` command:

    ```bash
    crawl4ai-setup
    ```

## Usage

1.  **Run the Streamlit application**:

    Navigate to the project directory in your terminal and run:

    ```bash
    streamlit run main.py
    ```

2.  **Access the application**: 

    Your web browser will automatically open to the Streamlit application (usually at `http://localhost:8501`).

3.  **Configure and Crawl**:

    -   Enter the target website URL in the sidebar.
    -   Adjust the crawl settings (Max Crawl Depth, Max Pages to Crawl, Strategy, etc.) as desired.
    -   Click the "Start Crawling" button to begin the process.

4.  **Download and Preview**:

    Once the crawl is complete, you will see options to download the content as a combined Markdown file or a ZIP archive. You can also preview the first crawled page.

## License

This project is licensed under the [LICENSE](LICENSE) file. Please see the file for more details.
