from concurrent.futures import ThreadPoolExecutor
import os
import logging
import warnings
import time
from api import gpt
from utils import file_parser, text_chunker

# Configure logging
logging.basicConfig(level=logging.INFO)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

def process_file(file):
    """
    Process a single file: parse, chunk, summarize, and ouput book summary.

    :param file: Path to the input file.
    :return: A summary of the book.
    """
    try:
        # Parse the file
        logging.info(f"Processing file: {file}")
        content = file_parser.ingest_file(file)

        # Maybe chunk the book
        chunks = text_chunker.chunk_text(content, force_chunking=True)

        summaries = []
        if len(chunks) == 1:
            summaries = chunks
        else:
            # Summarize each chunk
            start_time = time.time()
            summaries = gpt.generate_summaries_threaded(chunks)
            elapsed_time = time.time() - start_time
            logging.info(f"Summarized {len(chunks)} chunks in {elapsed_time:.2f} seconds")

        # Combine chunk summaries into a full book summary
        start_time = time.time()
        book_summary = gpt.summarize_book(summaries)
        elapsed_time = time.time() - start_time
        logging.info(f"Summarized full book in {elapsed_time:.2f} seconds")

        # Write the book summary to the output file
        with open("book_summaries.txt", "a", encoding="utf-8") as output:
            output.write(f"Summary of {file}\n")
            output.write(book_summary + "\n\n")

        return book_summary

    except Exception as e:
        logging.error(f"Error processing {file}: {e}")
        return None

if __name__ == "__main__":
    total_time = time.time()

    # Input files: read all files in the data/books directory
    data_dir = "data/books"
    test_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, f))
    ]

    book_summaries = []

    # Use ThreadPoolExecutor to process files in parallel
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_file, test_files))

    # Collect summaries
    book_summaries = [summary for summary in results if summary is not None]

    # Generate a comparative book report
    if book_summaries:
        start_time = time.time()
        comparative_report = gpt.generate_comparative_book_report(book_summaries)
        elapsed_time = time.time() - start_time
        logging.info(f"Generated final essay in {elapsed_time:.2f} seconds")

        with open("comparative_report.txt", "w", encoding="utf-8") as output:
            output.write(comparative_report)

    total_elapsed_time = time.time() - total_time
    logging.info(f"Total process completed {total_elapsed_time:.2f} seconds")
