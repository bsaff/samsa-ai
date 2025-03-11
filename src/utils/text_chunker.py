import tiktoken

def chunk_text(book_text, max_summary_tokens=2000, context_window=128_000, chunk_size=10_000, force_chunking=False):
    """
    Tokenizes an entire book for the specified model and splits it into chunks if it's too big.

    :param book_text: The full text of the book as a string.
    :param max_summary_tokens: Maximum response tokens reserved for book summary generation.
    :param context_window: The model's context window size.
    :param chunk_size: Chunk size for splitting.
    :param force_chunk: Chunk it regardless of context_window constraints.
    :return: A list of decoded text chunks
    """
    try:
        # Load the tokenizer for the specified model
        tokenizer = tiktoken.encoding_for_model('gpt-4o-mini')

        # Tokenize the book text
        tokens = tokenizer.encode(book_text)
        token_count = len(tokens)

        # Determine whether to chunk or process the entire book
        if token_count + max_summary_tokens <= context_window and not force_chunking:
            # Process the entire book as one chunk
            return [tokenizer.decode(tokens)]
        else:
            # Split tokens into chunks dynamically
            chunks = [tokens[i:(i + chunk_size)] for i in range(0, token_count, chunk_size)]

            # Decode each chunk back into text
            decoded_chunks = [tokenizer.decode(chunk) for chunk in chunks]

            return decoded_chunks

    except Exception as e:
        raise ValueError(f"Error tokenizing the book: {e}")
