import time
import uuid
import random
import threading
import os
import re
from pathlib import Path
from urllib.parse import urlparse
from crawl4ai.components.crawler_monitor import CrawlerMonitor
from crawl4ai.models import CrawlStatus

MAX_URLS = 10


################################ Utility Functions ################################


def sanitize_filename(url):
    """Create a safe filename from a URL."""
    parsed = urlparse(url)
    # Combine the domain and path to create a base filename
    filename = parsed.netloc + parsed.path
    # Remove any trailing slash to avoid empty file extensions
    filename = filename.rstrip("/")
    # Replace characters invalid for filenames with underscores
    filename = re.sub(r"[^\w\-_.]", "_", filename)
    # Truncate the filename if it exceeds a reasonable length
    if len(filename) > 200:
        filename = filename[:200]
    # Append the markdown file extension
    return f"{filename}.md"


def save_crawled_page(
    task_id,
    url,
    status,
    start_time,
    end_time,
    memory_usage,
    peak_memory,
    error_message=None,
    output_dir="crawled_pages",
):
    """Save only markdown-formatted content for the crawled page."""
    # Create the output directory if it doesn't already exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate a filesystem-safe filename from the URL
    filename = sanitize_filename(url)
    filepath = os.path.join(output_dir, filename)

    # Prepare clean markdown content based on crawl status
    if status == CrawlStatus.COMPLETED:
        # Save only the markdown content without metadata headers
        content = """# Example Page

            This is a simulated crawl from the CrawlerMonitor example.
            In a real implementation, this would contain the actual webpage content
            extracted as clean markdown, preserving the document structure and readability.

            ## Introduction

            This page was successfully crawled and processed. The content below represents
            what would typically be extracted from a real webpage.

            ## Main Content

            Lorem ipsum dolor sit amet, consectetur adipiscing elit. This is sample content
            that demonstrates how the actual webpage text would be preserved in markdown format.

            ### Key Features

            - Clean markdown formatting
            - Preserved document structure
            - No technical metadata in content
            - Easy to read and process

            ### Sample Data

            The page contains typical web content including:
            - Headings and subheadings
            - Paragraphs of text
            - Lists and bullet points
            - Structured information

            ## Conclusion

            This markdown file represents the extracted content from the webpage,
            formatted for easy reading and further processing.
            """
    elif status == CrawlStatus.FAILED:
        # For failed crawls, include minimal error information in markdown
        content = f"""# Crawl Failed

        **Error:** {error_message or "Unknown error"}

            The crawl attempt for this page was unsuccessful.
            """
    else:
        # For other statuses, provide minimal information
        content = f"""# Crawl Status: {status.name}

        This crawl task is currently in {status.name} state.
        """

    # Write the clean markdown content to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath


################################ Simulation Logic ################################


def simulate_webcrawler_operations(monitor, num_tasks=20):
    """Simulate a web crawler's operations with multiple concurrent tasks."""
    print(f"Starting simulation with {num_tasks} tasks...")

    # Pre-register all tasks with the monitor before starting processing
    task_ids = []
    for _ in range(num_tasks):
        task_id = str(uuid.uuid4())
        url = "https://www.promptingguide.ai/"
        monitor.add_task(task_id, url)
        task_ids.append((task_id, url))
        # Introduce a small delay to simulate tasks arriving over time
        time.sleep(0.2)

    # Create a worker thread for each task to simulate concurrent processing
    threads = []
    for i, (task_id, url) in enumerate(task_ids):
        thread = threading.Thread(target=process_task, args=(monitor, task_id, url, i))
        thread.daemon = True  # Allow main program to exit even if threads are running
        threads.append(thread)

    # Start threads in batches to control the level of concurrency
    batch_size = 4  # Process 4 tasks at a time
    for i in range(0, len(threads), batch_size):
        batch = threads[i : i + batch_size]
        # Start each thread in the current batch
        for thread in batch:
            thread.start()
            time.sleep(0.5)  # Stagger thread start times to avoid thundering herd

        # Wait a moment before starting the next batch
        time.sleep(random.uniform(1.0, 3.0))

        # Refresh the queue statistics in the monitor UI
        update_queue_stats(monitor)

        # Simulate changes in system memory pressure based on active threads
        active_threads = [t for t in threads if t.is_alive()]
        if len(active_threads) > 8:
            monitor.update_memory_status("CRITICAL")
        elif len(active_threads) > 4:
            monitor.update_memory_status("PRESSURE")
        else:
            monitor.update_memory_status("NORMAL")

    # Block until all simulation threads have finished their execution
    for thread in threads:
        thread.join()

    # Perform final updates after all tasks are done
    update_queue_stats(monitor)
    monitor.update_memory_status("NORMAL")

    print("Simulation completed!")


def process_task(monitor, task_id, url, index):
    """Simulate the lifecycle of processing a single crawl task."""
    # Simulate the time a task spends waiting in the queue before processing
    wait_time = random.uniform(0.5, 3.0)
    time.sleep(wait_time)

    # Mark the task as "IN_PROGRESS" and record the start time
    start_time = time.time()
    monitor.update_task(
        task_id=task_id,
        status=CrawlStatus.IN_PROGRESS,
        start_time=start_time,
        wait_time=wait_time,
    )

    # Simulate the task's processing work and fluctuating memory usage
    total_process_time = random.uniform(2.0, 10.0)
    step_time = total_process_time / 5  # Divide work into 5 update steps
    peak_memory = 0.0
    for step in range(5):
        # Simulate memory usage increasing and then decreasing
        if step < 3:  # First 3 steps: memory increases
            memory_usage = random.uniform(5.0, 20.0) * (step + 1)
        else:  # Last 2 steps: memory decreases
            memory_usage = random.uniform(5.0, 20.0) * (5 - step)

        # Track the highest memory usage seen for this task
        peak_memory = max(
            memory_usage, monitor.get_task_stats(task_id).get("peak_memory", 0)
        )

        # Update the monitor with the latest memory stats
        monitor.update_task(
            task_id=task_id, memory_usage=memory_usage, peak_memory=peak_memory
        )

        time.sleep(step_time)

    # Determine the final status of the task (success or failure)
    end_time = time.time()
    if index % 5 == 0:  # Make every 5th task fail for demonstration
        error_message = "Connection timeout"
        monitor.update_task(
            task_id=task_id,
            status=CrawlStatus.FAILED,
            end_time=end_time,
            memory_usage=0.0,
            error_message=error_message,
        )
        # Save a report for the failed page
        save_crawled_page(
            task_id=task_id,
            url=url,
            status=CrawlStatus.FAILED,
            start_time=start_time,
            end_time=end_time,
            memory_usage=0.0,
            peak_memory=peak_memory,
            error_message=error_message,
        )
    else:
        monitor.update_task(
            task_id=task_id,
            status=CrawlStatus.COMPLETED,
            end_time=end_time,
            memory_usage=0.0,
        )
        # Save a report for the successful page
        filepath = save_crawled_page(
            task_id=task_id,
            url=url,
            status=CrawlStatus.COMPLETED,
            start_time=start_time,
            end_time=end_time,
            memory_usage=0.0,
            peak_memory=peak_memory,
        )
        print(f"✓ Saved: {filepath}")


def update_queue_stats(monitor):
    """Calculate and update queue-wide statistics."""
    task_stats = monitor.get_all_task_stats()

    # Filter for tasks that are still in the QUEUED state
    queued_tasks = [
        stats
        for stats in task_stats.values()
        if stats["status"] == CrawlStatus.QUEUED.name
    ]
    total_queued = len(queued_tasks)

    # If there are tasks in the queue, calculate wait time statistics
    if total_queued > 0:
        current_time = time.time()
        # Calculate how long each queued task has been waiting
        wait_times = [
            current_time - stats.get("enqueue_time", current_time)
            for stats in queued_tasks
        ]
        highest_wait_time = max(wait_times, default=0.0)
        avg_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0.0
    else:
        highest_wait_time = 0.0
        avg_wait_time = 0.0

    # Push the calculated statistics to the monitor for display
    monitor.update_queue_statistics(
        total_queued=total_queued,
        highest_wait_time=highest_wait_time,
        avg_wait_time=avg_wait_time,
    )


################################ Main Execution ################################


def main():
    # Initialize the crawler monitor with specific UI configurations
    monitor = CrawlerMonitor(
        urls_total=MAX_URLS,  # Set the total number of URLs to be processed
        refresh_rate=0.5,  # Update the UI display twice per second
        enable_ui=True,  # Activate the terminal-based user interface
        # max_width=120,  # Limit the display width to 120 characters
    )

    # Start the monitor's rendering loop in a separate thread
    monitor.start()

    # Use a try/finally block to ensure the monitor is stopped gracefully
    try:
        # Run the main web crawling simulation
        simulate_webcrawler_operations(monitor)

        # Pause briefly to allow viewing the final state in the UI
        print("Waiting to view final state...")
        time.sleep(5)

    except KeyboardInterrupt:
        print("\nExample interrupted by user")
    finally:
        # Stop the monitor and clean up resources
        monitor.stop()
        print("Example completed!")

        # Retrieve and print a final summary of the crawl statistics
        summary = monitor.get_summary()
        print("\nCrawler Statistics Summary:")
        print(f"Total URLs: {summary['urls_total']}")
        print(f"Completed: {summary['urls_completed']}")
        print(f"Completion percentage: {summary['completion_percentage']:.1f}%")
        print(f"Peak memory usage: {summary['peak_memory_percent']:.1f}%")

        # Display the counts for each final task status
        status_counts = summary["status_counts"]
        print("\nTask Status Counts:")
        for status, count in status_counts.items():
            print(f"  {status}: {count}")

        print(f"\nAll crawled pages saved to 'crawled_pages/' directory")


if __name__ == "__main__":
    main()
