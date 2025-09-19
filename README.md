**Earnings Call Analyzer**

This is a Streamlit application designed to process earnings call transcript PDFs, extract key insights, generate business-focused topics and summaries, and provide a conversational Q&A interface.

Prerequisites
Python 3.8 or higher

A Google Cloud Project or access to the Gemini API for generative models.

An API key for Llama Parse to extract text from PDFs.

Getting Started
Follow these steps to set up your environment and run the application.

1. Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

For macOS and Linux:

```
python3 -m venv venv
source venv/bin/activate
```

For Windows:

```
python -m venv venv
venv\Scripts\activate
```

2. Configure Environment Variables

I have already given .env file 

You will need to add your API keys to this file. The application is configured to use the Google Gemini API and Llama Parse.

Google Gemini API through the Google AI Studio 

LLama Parse API Key through [LlamaCloud](https://developers.llamaindex.ai/python/cloud/api_key)

3. Install Dependencies
With your virtual environment activated, install all the necessary libraries from the requirements.txt file.

```
pip install -r requirements.txt
```
4. Place the Sample File
Make sure the sample earnings call transcript PDF file, Q2FY24_LaurusLabs_EarningsCallTranscript.pdf, is located in the same directory as your application's main Python script (e.g., App.py).

5. Run the Application
Once all dependencies are installed and the environment variables are set, you can run the Streamlit application.

```
streamlit run App.py
```

The application will open in your web browser. You are now ready to upload the PDF, analyze the transcript, and interact with the chatbot.

**Technical Aspects**
The application is built on a robust Retrieval-Augmented Generation (RAG) architecture, leveraging several key components to deliver its core functionality.

1. PDF Extraction and Structuring:
Tool: Llama Parse is used for high-fidelity extraction of text from PDF documents. It is particularly effective at handling complex, multi-column layouts often found in financial reports.

Process: The raw text is extracted and then structured into a format that distinguishes between speakers and their messages. fuzzywuzzy and python-Levenshtein are employed to intelligently identify and match speaker names, ensuring a clean and organized transcript.

2. Vectorization and Embedding:
Tool: sentence-transformers and HuggingFaceEmbeddings are used to convert the processed text chunks into numerical vectors. These vectors capture the semantic meaning of the text.

Process: Each meaningful chunk of the transcript (e.g., a speaker's answer) is transformed into a high-dimensional vector. This process is crucial for enabling semantic search, allowing the system to find relevant content based on meaning rather than just keywords.

3. Vector Store and Retrieval:
Tool: FAISS-CPU acts as the vector database. It is a highly optimized library for efficient similarity search.

Process: All the vectorized text chunks are indexed in the FAISS store. When a user asks a question, their query is also converted into a vector. FAISS then quickly finds the most semantically similar chunks from the transcript, retrieving only the most relevant context. This is the "Retrieval" part of RAG.

4. Generative AI and Response Synthesis:
Tool: google-generative-ai is the large language model (LLM) that powers the final response generation. It is integrated via langchain.

Process: The LLM receives the user's original question along with the relevant text chunks retrieved from the FAISS vector store. The model then synthesizes this information to provide a concise, accurate, and contextually-aware answer. This is the "Generation" part of RAG.

5. User Interface:
Tool: Streamlit provides the user-friendly web interface, handling the file upload, progress indicators, and chatbot interactions.

Future Enhancements
The current application serves as a strong foundation. Here are several features that could be added to improve its robustness, accuracy, and user experience.

1. Adding Guardrails:
Concept: Implement a system to prevent the LLM from generating responses that are off-topic, hallucinated, or inappropriate.

Implementation: This could involve a separate LLM call to pre-check the user's query or post-check the model's response. For example, a "fact-checking" chain could be added to ensure the generated answer is grounded in the provided transcript.

2. Hybrid Retrieval:
Concept: Combine the strengths of both semantic search and keyword-based search.

Implementation: Use a library like BM25 for keyword matching in addition to the existing FAISS for semantic similarity. A Hybrid Retriever would then combine the results from both methods, ensuring that even if a concept isn't semantically similar, an exact keyword match is still considered.

3. Chat History and Conversation Memory:
Concept: Allow the chatbot to remember previous turns in the conversation to provide more natural and coherent answers.

Implementation: Store the chat history in the Streamlit session state and pass a summary of the conversation to the LLM with each new query. This would enable follow-up questions like, "What about their CDMO business?" without needing to repeat the full context.

4. Multi-Document Analysis:
Concept: Allow users to upload multiple transcripts and perform comparative analysis.

Implementation: The vector store could be extended to include documents from different quarters or years. This would enable queries like, "How did the revenue from Q2 FY24 compare to Q2 FY23?"

