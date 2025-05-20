from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import time
from openai import OpenAI
import re

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def extract_retry_ms(log: str):
    if "rate_limit_exceeded" in log:
        match = re.search(r'please try again in (\d+)ms', log, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None

def summarize_chunk(chunk, max_retries=3):
    """
    Summarizes a single chunk of a book using the Chat API.
    Retries if rate limit is hit.

    :param chunk: The text chunk to summarize.
    :param max_retries: Maximum number of retries before giving up.
    :return: Summary of the chunk or an error message.
    """
    attempt = 0
    while attempt <= max_retries:
        try:
            logging.info(f"Summarizing chunk... Attempt {attempt + 1}")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant summarizing parts of a book. Focus on capturing the key points, events, and details accurately and factually. Do not provide analysis or commentary."
                    },
                    {
                        "role": "user",
                        "content": f"Summarize the following text. Provide an objective overview of the events, characters, and significant details described in the passage. Avoid interpretation, commentary, or thematic analysis.\n\n{chunk}"
                    }
                ],
                max_tokens=2000,
                temperature=0.7
            )
            return response.choices[0].message.content

        except Exception as e:
            log_msg = str(e)
            logging.error(f"Error summarizing chunk: {log_msg}")

            retry_ms = extract_retry_ms(log_msg)
            if retry_ms:
                time.sleep(retry_ms / 1000)
                attempt += 1
            else:
                break

    return f"Error summarizing chunk after {max_retries} retries."

def generate_summaries_threaded(chunks):
    """
    Summarizes chunks concurrently.

    :param chunks: List of text chunks to summarize.
    :return: List of summaries.
    """

    with ThreadPoolExecutor() as executor:
        # Submit tasks to the thread pool
        futures = {executor.submit(summarize_chunk, chunk): i for i, chunk in enumerate(chunks)}
        summaries = []

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                summaries.append(future.result())
            except Exception as e:
                logging.error(f"Error processing chunk: {e}")
                summaries.append(f"Error processing chunk: {e}")

    return summaries


def summarize_book(summaries):
    """
    Use partial book summaries to create a full book summary.

    :param summaries: List of summary strings, one per chunk.
    :return: A single string representing the cohesive book-wide summary.
    """
    combined_summaries = "\n\n".join(summaries)

    try:
        logging.info("Generating book-wide summary...")

        # Use GPT to create a single cohesive summary
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant tasked with summarizing a book. Focus on crafting a cohesive and thematic summary based on provided text. The theme of social isolation should be emphasized in your final summary."
                },
                {
                    "role": "user",
                    "content": f"Read the following text and output a comprehensive summary of the novel. Ensure the summary includes vivid, specific events, symbols, and imagery that emphasize the theme of social isolation. Avoid redundant details.\n\n{combined_summaries}"
                }
            ],
            max_tokens=2000,
            temperature=0.7
        )

        # Extract the generated summary
        book_summary = response.choices[0].message.content
        return book_summary

    except Exception as e:
        logging.error(f"Error generating book-wide summary: {e}")
        return None

def generate_comparative_book_report(book_summaries):
    """
    Generates a comparative book report based on the summaries of multiple books.

    :param book_summaries: List of summaries for each book.
    :return: A 5-paragraph comparative book report as a string.
    """

    # Combine all book summaries into a single input string
    combined_summaries = "\n\n".join([f"Book Summary:\n{summary}" for summary in enumerate(book_summaries)])

    try:
        logging.info("Generating comparative book report...")

        # Use GPT to create a comparative book report
        response = client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant tasked with writing a five paragraph comparative literature essay. Construct a clear thesis statement, present arguments based on the content of each novel, and providing a concluding paragraph to summarize the arguments."
                },
                {
                    "role": "user",
                    "content": f"Using the following book summaries, write a five paragraph comparative literature essay. Focus on analyzing how each book addresses the theme of social isolation, presenting a clear thesis, providing interesting intertextual comparisons, and concluding with a synthesis of the analysis.\n\n{combined_summaries}"
                }
            ],
            temperature=0.7
        )

        # Extract the generated report
        book_report = response.choices[0].message.content

        return book_report

    except Exception as e:
        logging.error(f"Error generating book report: {e}")
        return None
