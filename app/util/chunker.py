"""
chunker.py
Token-based and sentence-aware text chunking for legal document RAG systems.
"""

import nltk
from nltk.tokenize import sent_tokenize
from typing import List

# Download NLTK tokenizer if not already present
nltk.download("punkt", quiet=True)


def chunk_text_sentence_based(text: str, max_words: int = 200) -> List[str]:
    """
    Basic sentence-aware chunking.
    Useful fallback when tokenizer is unavailable.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        length = len(words)

        if current_len + length > max_words:
            chunks.append(" ".join(current_chunk).strip())
            current_chunk = []
            current_len = 0

        current_chunk.append(sentence)
        current_len += length

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks


def chunk_text_token_based(
    text: str,
    tokenizer,
    max_tokens: int = 256,
    overlap: int = 20
) -> List[str]:
    """
    Token-based, sentence-aware chunking.
    Best for RAG systems using embeddings or LLMs.

    Params:
        text: full text to chunk
        tokenizer: HuggingFace tokenizer instance
        max_tokens: maximum tokens per chunk
        overlap: number of tokens to repeat (sliding window)

    Returns:
        List[str]: list of chunk strings
    """

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Tokens for current chunk
        current_tokens = tokenizer.encode(
            current_chunk,
            add_special_tokens=False
        )

        # Tokens for new sentence
        sentence_tokens = tokenizer.encode(
            sentence,
            add_special_tokens=False
        )

        # If adding this sentence exceeds limit → finalize current chunk
        if len(current_tokens) + len(sentence_tokens) > max_tokens:
            chunks.append(current_chunk.strip())

            # Build overlap
            if overlap > 0 and len(current_tokens) > overlap:
                overlap_tokens = current_tokens[-overlap:]
                current_chunk = tokenizer.decode(overlap_tokens)
            else:
                current_chunk = ""

        # Add sentence to chunk
        current_chunk += " " + sentence

    # Add last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def smart_chunker(
    text: str,
    tokenizer=None,
    max_tokens: int = 256,
    overlap: int = 20,
    fallback_max_words: int = 200
) -> List[str]:
    """
    Automatically selects BEST chunking strategy.
    If tokenizer is provided → use token-based.
    Else → fallback to sentence-based.

    Params:
        text: input text
        tokenizer: optional HF tokenizer
        max_tokens: token limit for chunks
        overlap: overlap tokens
        fallback_max_words: word limit for sentence-based chunking
    """

    if tokenizer:
        return chunk_text_token_based(
            text,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            overlap=overlap
        )

    return chunk_text_sentence_based(text, max_words=fallback_max_words)

