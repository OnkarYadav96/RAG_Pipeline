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
