This project is a backend-first legal document summarization pipeline designed to process long legal PDFs and generate grounded, concise summaries using Large Language Models (LLMs).
The system is built with a Retrieval-Augmented Generation (RAG)-ready architecture, enabling future extension into document question-answering and agent-based workflows.

## Problem Statement
Legal documents are often lengthy, complex, and time-consuming to analyze.
Traditional summarization approaches struggle with context retention and hallucinations when handling long documents.
This project addresses these challenges by:
* Structuring documents into semantic chunks
* Preparing data for retrieval-based grounding
* Constraining LLM outputs to relevant document context

## Key Features
* PDF ingestion and text extraction
* Semantic chunking for long documents
* Embedding-based document representation
* LLM-driven summarization with reduced hallucinations
* Modular backend design (CLI + API-ready)
* Architecture ready for RAG-based document Q&A

## Architecture
![Architecture]([Legal Document Summarizer — Full RAG Architecture (Free Stack) (2).png](https://github.com/Purushottam29/Legal_Doc_Summarizer/blob/76f44dc8d4d4c20440143b9e8c58fdfe1e9123c0/Legal%20Document%20Summarizer%20%E2%80%94%20Full%20RAG%20Architecture%20(Free%20Stack)%20(2).png))

## Tech Stack
* Language: Python
* Backend: FastAPI
* LLM: Ollama(Mistral)
* Vector Search: ChromaDB
* Document Processing: PDF parsing & text extraction

## Project Structure
## How it works
* Legal PDFs are ingested and parsed into raw text
* Text is divided into semantically meaningful chunks
* Chunks are embedded and stored for retrieval
* Relevant content is passed to the LLM for grounded summarization
* The output summary is generated with minimized hallucination risk

## How to run locally
* Downlaod and install Ollama mistral
* Clone the repo
* Install the requirements.txt file in your virtual environment

  ```bash
  python -m venv LDS
  source LDS/bin/activate
  pip install -r requirements.txt
  
* Use swagger ui to test the APIs. Run this in your terminal
  
  ```bash
  uvicorn app.main:app --reload
  
## Limitations
* Retrieval quality depends on chunking strategy
* Latency increases for very large documents
* No automated evaluation metrics implemented yet

## Future Enhancements
* Full Retrieval-Augmented Question Answering (RAG)
* Automated evaluation using precision and groundedness metrics
* Web dashboard integration

## Author
### Purushottam Choudhary
B.Tech Computer Science
* Github: https://github.com/Purushottam29
* LinkedIN: https://www.linkedin.com/in/purushottam-choudhary-166120373
* Mail: purushottamchoudhary2910@gmail.com
