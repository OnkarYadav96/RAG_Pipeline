Optimize RAG Chatbot
Upload PDF or TXT files, then ask questions grounded in your documents. Uses local embeddings + Groq LLM (API key from .env).

***** DATA INJECTION *****
 Load all the Documents(PDF,Texts) files (DOCUMENT DATA STRUCTURE) ==> Using Documents Loaders (PDFLoader,WebLoader,TextLoader)
 Create Chaunk (with Overlappig to keep meaning in Chunking) ==> While Chinking add metadata (user_id,dates,authers,owners) for easy retriaval
 Create Embeddings (using Setence Transformer) ==>Chunk into Vector Format
 Vector DB ==> (create a local Persisit_directory in Application for  data) store data in Vector DB
               While storing Vectors in VectorDB we also pass Documents list (Which Document having how many embeddings)
			   
****** RETRIAVAL PROCESS *****
 Convert your Query into Embedding first
 In Retrival function we pass Vector_Store,Embeddings_function(Because that also need to convert Query into Embedding vector)
 Retriaval method having (query,top_k,threshould_score) 
 Search in vector store==> vector_store.collection.query(query,top_k_result)
 Retrival function return ARGUMENTED CONTEXT DATA
 
******* LLM Search ******
 Provide your Augumented data + Query into LLM Model
 Rertive top_k results and rearange the results

  
USE FOLLOWING KEYWORDS
(Streaming,Batch, Structured Output(Pydantic,TypeDict),Middleswares (Summurization,Human-in-the-loop,Model call limit,Token call limit) e.g triggers=message,tokens) 
LangGraph (Nodes,Edges,States(change state use Reducers)) (TOOLS,MCP Servers), Add memory


Vectorless RAG pipeline — official PageIndex SDK workflow.

Reference: https://github.com/VectifyAI/PageIndex

PageIndex replaces vector DB + chunking with:
  Step 1 — Index:  build a hierarchical tree index per document
  Step 2 — Retrieve: LLM reasons over the tree (not similarity search)
  Step 3 — Generate: answer from text in selected tree nodes

Compare with your vector RAG in notebook/pdf_loader.ipynb:
  Vector RAG:  chunk → embed → Chroma → cosine similarity → LLM
  PageIndex:   tree  → LLM tree search → fetch nodes → LLM



  ### Chatbot And RAG Evaluation

Retrieval Augmented Generation (RAG) is a technique that enhances Large Language Models (LLMs) by providing them with relevant external knowledge. It has become one of the most widely used approaches for building LLM applications.

This tutorial will show you how to evaluate your RAG applications using LangSmith. You'll learn:

1. How to create test datasets
2. How to run your RAG application on those datasets
3. How to measure your application's performance using different evaluation metrics

#### Overview
A typical RAG evaluation workflow consists of three main steps:

1. Creating a dataset with questions and their expected answers
2. Running your RAG application on those questions
3. Using evaluators to measure how well your application performed, looking at factors like:
 - Answer relevance
 - Answer accuracy
 - Retrieval quality
