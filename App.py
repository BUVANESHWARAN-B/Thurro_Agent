import streamlit as st
import os
import re
import json
import tempfile
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv

# Third-party libraries for parsing and structuring
from fuzzywuzzy import process
from llama_parse import LlamaParse

# LangChain libraries for RAG
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Load Environment Variables ---
load_dotenv()
# Ensure you have LLAMA_CLOUD_API_KEY and GOOGLE_API_KEY in your .env file
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Pydantic Models for Structured LLM Output ---

class ManagementParticipant(BaseModel):
    """Data model for a single management participant."""
    name: str = Field(description="The full name of the management participant.")
    designation: str = Field(description="The job title or designation of the participant.")

class CallMetadata(BaseModel):
    """Data model for the entire call's metadata."""
    company_name: str = Field(description="The name of the company.")
    call_date: str = Field(description="The date of the conference call.")
    management: List[ManagementParticipant] = Field(
        description="A list of all management participants on the call."
    )
    
class TopicList(BaseModel):
    """List of key business topics extracted from a financial transcript section."""
    topics: List[str] = Field(
        description="A list of 3-5 key business-relevant topics from the text."
    )

class TopicSummary(BaseModel):
    """A 2-4 sentence summary for a specific topic, with key data points."""
    summary: str = Field(
        description="A concise summary of the topic from the transcript, including relevant numbers and insights."
    )


# --- Component 1: Core Processing and Structuring Functions ---

def delineate_transcript_two_sections_generic(file_content: str):
    q_and_a_start_pattern = re.compile(
        r"we will now begin the question-and-answer session"
        r"|"
        r"(?:we'll|we will) take the first question from the line of",
        re.IGNORECASE | re.DOTALL
    )
    q_and_a_start_match = q_and_a_start_pattern.search(file_content)

    if q_and_a_start_match:
        q_and_a_start_index = q_and_a_start_match.start()
        line_start_index = file_content.rfind('\n', 0, q_and_a_start_index) + 1
        opening_remarks = file_content[:line_start_index].strip()
        q_and_a_session = file_content[line_start_index:].strip()
        return opening_remarks, q_and_a_session
    else:
        st.warning("Could not find the Q&A start marker.")
        return file_content, ""

def create_speaker_chunks(section_text: str, section_type: str, management_list: List[ManagementParticipant]):
    processed_text = "\n\n" + section_text.strip()
    management_names = [p.name for p in management_list]
    speaker_pattern = re.compile(r"\n\n#?\s*([A-Za-z.\s]+?):", re.DOTALL)
    parts = speaker_pattern.split(processed_text)
    chunks = []
    
    for i in range(1, len(parts), 2):
        speaker_raw = parts[i].strip()
        message = parts[i+1].strip()
        match, score = process.extractOne(speaker_raw, management_names)
        speaker_role = "Analyst/Investor"
        speaker_name = speaker_raw
        if score > 85:
            speaker_role = "Management"
            speaker_name = match
        elif "moderator" in speaker_raw.lower():
            speaker_role = "Moderator"
            speaker_name = "Moderator"
        if message:
            chunks.append({
                "name": speaker_name, "role": speaker_role,
                "section": section_type, "message": message
            })
    return chunks

def tag_and_match_qa(qa_chunks: list, metadata: CallMetadata):
    qa_pairs = []
    current_question = None
    last_analyst_company = ' ' # New variable to hold the last company seen

    for i, chunk in enumerate(qa_chunks):
        if chunk['role'] == "Moderator":
            # A moderator signals a new question is coming.
            if i + 1 < len(qa_chunks) and qa_chunks[i + 1]['role'] == "Analyst/Investor":
                if current_question and current_question.get('answers'):
                    qa_pairs.append(current_question)

                analyst_chunk = qa_chunks[i + 1]
                match = re.search(r"from the line of (.*?) from (.*?)(?:\.|\s*Please go ahead)", chunk['message'], re.IGNORECASE)
                analyst_name = analyst_chunk.get('name', 'N/A')
                company = 'N/A'
                if match:
                    analyst_name = match.group(1).strip()
                    company = match.group(2).strip()
                    last_analyst_company = company # Update the last known company

                current_question = {
                    "question": {
                        "name": analyst_name,
                        "company": company,
                        "qa_tag": "question",
                        "message": analyst_chunk['message']
                    },
                    "answers": []
                }
            
        elif chunk['role'] == "Analyst/Investor":
            # Handles follow-up questions or missed moderator intros.
            if current_question and current_question.get('answers'):
                qa_pairs.append(current_question)
                
            current_question = {
                "question": {
                    "name": chunk.get('name', 'N/A'),
                    "company": last_analyst_company, # Use the last known company
                    "qa_tag": "question",
                    "message": chunk['message']
                },
                "answers": []
            }
            
        elif chunk['role'] == "Management" and current_question:
            designation = next((p.designation for p in metadata.management if p.name == chunk['name']), "")
            current_question['answers'].append({
                "name": chunk['name'],
                "designation": designation,
                "message": chunk['message'],
                "qa_tag": "answer"
            })
    
    # Append the last pair if it exists
    if current_question and current_question.get('answers'):
        qa_pairs.append(current_question)
    
    return qa_pairs

@st.cache_data(show_spinner="Processing PDF and structuring data...")
def process_transcript(file_content: str) -> Dict[str, Any]:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY)
    start_marker = "Moderator:"
    start_index = file_content.find(start_marker)
    if start_index == -1:
        st.error("Could not find the 'Moderator:' start marker in the transcript.")
        return {}
    
    metadata_content = file_content[:start_index].strip()
    transcript_content = file_content[start_index:].strip()
    opening_remarks_text, q_and_a_session_text = delineate_transcript_two_sections_generic(transcript_content)
    
    structured_llm = llm.with_structured_output(CallMetadata)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at extracting information... Extract the company name, date, and a list of all management participants."),
        ("human", "Extract the required information from this text: \n\n{metadata_text}")
    ])
    extraction_chain = prompt | structured_llm
    metadata_object = extraction_chain.invoke({"metadata_text": metadata_content})
    
    opening_remarks_chunks = create_speaker_chunks(opening_remarks_text, "Opening Remarks", metadata_object.management)
    raw_qa_chunks = create_speaker_chunks(q_and_a_session_text, "Q&A", metadata_object.management)
    
    # The function now returns one list of pairs
    qa_pairs = tag_and_match_qa(raw_qa_chunks, metadata_object)
    
    # Flatten the chunks again just for the vector store
    question_chunks_flat = [p['question'] for p in qa_pairs if p['question'].get('message')]
    answer_chunks_flat = [ans for p in qa_pairs for ans in p['answers']]

    return {
        "metadata": metadata_object,
        "opening_remarks_text": opening_remarks_text,
        "qa_session_text": q_and_a_session_text,
        "opening_remarks_chunks": opening_remarks_chunks,
        "qa_pairs": qa_pairs, # Store the new paired structure for the UI
        "question_chunks_flat": question_chunks_flat, # Keep flattened for RAG
        "answer_chunks_flat": answer_chunks_flat
    }


# --- Component 2: Topic and Summary Generation Functions ---

@st.cache_data(show_spinner="Extracting topics with AI...")
def generate_topics(_llm, section_text: str, section_type: str) -> List[str]:
    topic_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert financial analyst... extract 3-5 key business-relevant topics... Return a JSON list of strings."),
        ("human", "Extract topics from the {section_type} section: \n\n{text}")
    ])
    topic_extraction_chain = topic_prompt | _llm.with_structured_output(TopicList)
    topics_obj = topic_extraction_chain.invoke({"section_type": section_type, "text": section_text})
    return topics_obj.topics

@st.cache_data(show_spinner="Generating summary with AI...")
def generate_summary(_llm, topic: str, section_text: str) -> str:
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert financial analyst... create a 2-4 sentence summary for the topic '{topic}'... Return a JSON object with a single 'summary' key."),
        ("human", "Summarize the topic based on this text: \n\n{text}")
    ])
    summary_chain = summary_prompt | _llm.with_structured_output(TopicSummary)
    summary_obj = summary_chain.invoke({"topic": topic, "text": section_text})
    return summary_obj.summary


# --- Component 3: AI Assistant (RAG) Functions ---

def create_documents_for_retrieval(processed_data: Dict[str, Any]) -> List[Document]:
    """Prepares all chunks for ingestion into the vector store."""
    all_chunks = (
        processed_data["opening_remarks_chunks"] +
        processed_data["question_chunks_flat"] +
        processed_data["answer_chunks_flat"]
    )
    documents = []
    for i, chunk in enumerate(all_chunks):
        content_prefix = ""
        if chunk.get("section") == "Opening Remarks":
            content_prefix = f"Section: Opening Remarks, Speaker: {chunk['name']} ({chunk['role']})"
        elif chunk.get("qa_tag") == "question":
            content_prefix = f"Section: Q&A, Speaker: {chunk['name']} ({chunk.get('company', 'Analyst')}), Question"
        elif chunk.get("qa_tag") == "answer":
            content_prefix = f"Section: Q&A, Speaker: {chunk['name']} (Management), Answer"
        
        page_content = f"{content_prefix}:\n{chunk['message']}"
        
        metadata = {
            "source_id": i,
            "speaker": chunk.get('name', 'N/A'),
            "section": chunk.get('section', chunk.get('qa_tag', 'N/A')),
            "role": chunk.get('role', 'N/A')
        }
        documents.append(Document(page_content=page_content, metadata=metadata))
    return documents

@st.cache_resource(show_spinner="Initializing AI Assistant...")
def create_faiss_vector_store(_documents: List[Document]):
    """Creates and caches the FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-small-v2",
        model_kwargs={"trust_remote_code": True}
    )
    vector_store = FAISS.from_documents(_documents, embeddings)
    return vector_store

def format_docs(docs: List[Document]) -> str:
    """Formats retrieved documents for the LLM prompt."""
    return "\n\n".join(f"Source {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs))

def calculate_confidence(scores: List[float]) -> str:
    if not scores:
        return "Low"
    avg_score = np.mean(scores)
    if avg_score > 0.6: return "High"
    if avg_score > 0.45: return "Medium"
    return "Low"

# --- Streamlit UI Rendering Functions ---

def render_welcome_page():
    st.header("Welcome & Upload")
    st.write("Upload a PDF transcript to begin analysis, or load the demo transcript to see the app's capabilities.")
    
def render_opening_remarks_page():
    st.header("Opening Remarks Analysis")
    data = st.session_state.data
    llm = st.session_state.llm
    
    tab1, tab2, tab3 = st.tabs(["📑 View Chunks", "📊 Generate Topics", "✍️ Summaries"])
    with tab1:
        st.write("Full transcript chunks for the Opening Remarks section:")
        for chunk in data["opening_remarks_chunks"]:
            with st.expander(f"**{chunk.get('name', 'N/A')}** ({chunk.get('role', 'N/A')})"):
                st.write(chunk.get('message', ''))
    with tab2:
        if st.button("Generate Topics for Opening Remarks"):
            st.session_state.opening_remarks_topics = generate_topics(llm, data["opening_remarks_text"], "Opening Remarks")
        if st.session_state.opening_remarks_topics:
            st.success("Topics have been generated!")
            for topic in st.session_state.opening_remarks_topics:
                st.markdown(f"- {topic}")
    with tab3:
        if st.session_state.opening_remarks_topics:
            selected_topic = st.selectbox("Select a topic to summarize:", st.session_state.opening_remarks_topics)
            if st.button("Generate Summary", key="or_summary"):
                summary = generate_summary(llm, selected_topic, data["opening_remarks_text"])
                st.session_state.opening_remarks_summaries[selected_topic] = summary
            if selected_topic in st.session_state.opening_remarks_summaries:
                st.info(f"Summary for: **{selected_topic}**")
                st.write(st.session_state.opening_remarks_summaries[selected_topic])
        else:
            st.warning("Go to the 'Generate Topics' tab first.")

def render_qa_session_page():
    st.header("Q&A Session Analysis")
    data = st.session_state.data
    llm = st.session_state.llm

    tab1, tab2, tab3 = st.tabs(["📑 View Chunks", "📊 Generate Topics", "✍️ Summaries"])

    with tab1:
        st.write("Full transcript chunks for the Q&A session, displayed as conversational exchanges:")

        # Loop through each question-and-answer pair
        for i, pair in enumerate(data["qa_pairs"]):
            q_chunk = pair["question"]
            
            # Display the question in a collapsible expander
            with st.expander(f"❓ Question {i+1}: **{q_chunk.get('name', 'N/A')}** ({q_chunk.get('company', 'N/A')})"):
                st.write(q_chunk.get('message', ''))
            
            # Display all answers associated with that question in their own expanders
            for a_chunk in pair["answers"]:
                with st.expander(f"🗣️ **{a_chunk.get('name', 'N/A')}** ({a_chunk.get('designation', 'Management')})"):
                    st.write(a_chunk.get('message', ''))
            st.markdown("---") # Separator between exchanges

    with tab2:
        if st.button("Generate Topics for Q&A Session"):
            st.session_state.qa_session_topics = generate_topics(llm, data["qa_session_text"], "Q&A Session")
        if st.session_state.qa_session_topics:
            st.success("Topics have been generated!")
            for topic in st.session_state.qa_session_topics:
                st.markdown(f"- {topic}")

    with tab3:
        if st.session_state.qa_session_topics:
            selected_topic = st.selectbox("Select a topic to summarize:", st.session_state.qa_session_topics)
            if st.button("Generate Summary", key="qa_summary"):
                summary = generate_summary(llm, selected_topic, data["qa_session_text"])
                st.session_state.qa_session_summaries[selected_topic] = summary
            if selected_topic in st.session_state.qa_session_summaries:
                st.info(f"Summary for: **{selected_topic}**")
                st.write(st.session_state.qa_session_summaries[selected_topic])
        else:
            st.warning("Go to the 'Generate Topics' tab first.")

def render_ai_assistant_page():
    st.header("AI Assistant")
    st.info("Ask specific questions about the call. If you ask the same question twice, the answer will be retrieved instantly from the cache.")

    data = st.session_state.data
    llm = st.session_state.llm
    metadata = data["metadata"]
    
    st.markdown(f"**Company:** {metadata.company_name} | **Call Date:** {metadata.call_date}")

    vector_store = st.session_state.vector_store
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.4, 'k': 5})

    rag_prompt = ChatPromptTemplate.from_template(
        """You are an expert financial AI assistant. Your task is to answer questions based ONLY on the provided earnings call transcript context.
- Be precise and factual.
- If the answer is not in the context, say "The answer is not available in the provided transcript."
- When quoting numbers or statements, mention the speaker if possible (e.g., "According to Dr. Satyanarayana Chava...").

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
    )

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    with st.expander("Sample Questions"):
        st.write("Click any question below to try it out:")
        cols = st.columns(2)
        sample_questions = [
            "What was the revenue growth this quarter?", "What are the key challenges mentioned by management?",
            "Any updates on new product launches?", "What is the company's outlook for the next quarter?",
            "Who are the key management personnel?", "What were the main financial highlights?",
            "Any discussion about market competition?", "What are the company's future plans?"
        ]
        for i, q in enumerate(sample_questions):
            if cols[i % 2].button(q, key=f"sample_{i}"):
                st.session_state.user_query = q
                st.rerun() # Use rerun to update the text input immediately
    
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    
    query = st.text_input("Type your question here:", value=st.session_state.user_query, key="query_input")

    col1, col2 = st.columns([1, 6])
    if col1.button("Ask Question"):
        if query:
            # Check if the answer is already in our cache
            if query in st.session_state.answer_cache:
                st.toast("Answer retrieved from cache!")
                cached_result = st.session_state.answer_cache[query]
                # Add the cached result to the top of the history
                st.session_state.chat_history.insert(0, cached_result)
            else:
                # If not in cache, run the full RAG chain
                with st.spinner("Searching transcript and generating new answer..."):
                    result = rag_chain_with_source.invoke(query)
                    
                    # Ensure context is a list of Document objects before getting scores
                    context_docs = result.get("context", [])
                    if isinstance(context_docs, list) and all(isinstance(doc, Document) for doc in context_docs):
                         # FAISS with similarity_score_threshold returns a tuple (doc, score)
                        scores = [doc.metadata.get('score', 0.0) for doc in context_docs]
                    else: # Handle cases where context might be formatted differently
                        scores = []
                    
                    confidence = calculate_confidence(scores)
                    
                    full_result = {
                        "query": query, 
                        "answer": result["answer"],
                        "context": result["context"], 
                        "confidence": confidence
                    }
                    # Save the new result to the cache and history
                    st.session_state.answer_cache[query] = full_result
                    st.session_state.chat_history.insert(0, full_result)

            st.session_state.user_query = ""
            st.rerun() # Rerun to clear the input box and show the new history item

    if col2.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.user_query = ""
        st.rerun()

    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("Conversation History")
        # Add 'exchange_idx' from enumerate to make each key unique
        for exchange_idx, exchange in enumerate(st.session_state.chat_history):
            st.markdown(f"<p style='background-color:#d9eaf7; padding:10px; border-radius:5px; color:black;'><b>You:</b> {exchange['query']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='background-color:#e6f2e6; padding:10px; border-radius:5px; color:black;'><b>AI Assistant ({exchange['confidence']} Confidence):</b> {exchange['answer']}</p>", unsafe_allow_html=True)
            with st.expander(f"View Sources ({len(exchange['context'])} chunks used)"):
                for i, doc in enumerate(exchange['context']):
                    score = doc.metadata.get('score', 'N/A')
                    if isinstance(score, float): score = f"{score:.3f}"
                    
                    # The updated, unique key:
                    unique_key = f"doc_{exchange_idx}_{i}_{exchange['query']}"
                    
                    st.markdown(f"**Source {i+1} | Relevance: {score}**")
                    st.text_area(
                        f"Chunk {doc.metadata.get('source_id')}", 
                        value=doc.page_content, 
                        height=150, 
                        key=unique_key
                    )
            st.markdown("---")


# --- Main Application Logic ---

st.set_page_config(layout="wide", page_title="Earnings Call Analyzer")

# Initialize session state variables
if "data" not in st.session_state:
    st.session_state.data = None
if "current_view" not in st.session_state:
    st.session_state.current_view = "Welcome & Upload"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "opening_remarks_topics" not in st.session_state:
    st.session_state.opening_remarks_topics = None
if "qa_session_topics" not in st.session_state:
    st.session_state.qa_session_topics = None
if "opening_remarks_summaries" not in st.session_state:
    st.session_state.opening_remarks_summaries = {}
if "qa_session_summaries" not in st.session_state:
    st.session_state.qa_session_summaries = {}
if "answer_cache" not in st.session_state:
    st.session_state.answer_cache = {}


# --- Sidebar Navigation ---
with st.sidebar:
    st.title("Earnings Call Analyzer")
    st.markdown("---")

    # --- Step 1: Uploader ---
    st.header("Transcript Uploader")
    uploaded_file = st.file_uploader(
        "Upload your transcript PDF", 
        type=["pdf"], 
        help="Upload a file to begin analysis."
    )
    
    # --- Step 2: Processing Buttons ---
    col1, col2 = st.columns(2)
    
    if uploaded_file:
        if col1.button("Analyze Uploaded File", use_container_width=True, type="primary"):
            with st.spinner("Parsing uploaded PDF... This may take a moment."):
                try:
                    # Use a temporary file to handle the uploaded data
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_file_path = tmp_file.name
                    
                    # Initialize parser and load data
                    parser = LlamaParse(api_key=LLAMA_CLOUD_API_KEY, result_type="markdown", verbose=False,hide_footers=True,hide_headers=True,skip_diagonal_text=True,disable_ocr=True)
                    documents = parser.load_data(temp_file_path)

                    if not documents:
                        st.error("The uploaded PDF could not be parsed. It might be empty, corrupted, or an image-based PDF.")
                    else:
                        full_text = "".join(doc.text + "\n\n" for doc in documents)
                        st.session_state.data = process_transcript(full_text)
                        st.session_state.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY)
                        st.session_state.documents = create_documents_for_retrieval(st.session_state.data)
                        st.session_state.vector_store = create_faiss_vector_store(st.session_state.documents)
                        st.session_state.current_view = "Opening Remarks"
                        # Reset states for the new file
                        st.session_state.opening_remarks_topics = None
                        st.session_state.qa_session_topics = None
                        st.session_state.opening_remarks_summaries = {}
                        st.session_state.qa_session_summaries = {}
                        st.session_state.chat_history = []
                        st.rerun() # Force UI update after processing
                
                except Exception as e:
                    st.error(f"An error occurred during file processing: {e}")
                finally:
                    # Clean up the temporary file
                    if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

    if col2.button("Load Demo Transcript", use_container_width=True):
        pdf_path = "Q2FY24_LaurusLabs_EarningsCallTranscript.pdf"
        if os.path.exists(pdf_path):
            with st.spinner("Parsing Demo PDF..."):
                parser = LlamaParse(api_key=LLAMA_CLOUD_API_KEY, result_type="markdown", verbose=False,hide_footers=True,hide_headers=True,skip_diagonal_text=True,disable_ocr=True)
                documents = parser.load_data(pdf_path)
                full_text = "".join(doc.text + "\n\n" for doc in documents)
                
                # --- State Initialization ---
                st.session_state.data = process_transcript(full_text)
                st.session_state.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY)
                st.session_state.documents = create_documents_for_retrieval(st.session_state.data)
                st.session_state.vector_store = create_faiss_vector_store(st.session_state.documents)
                st.session_state.current_view = "Opening Remarks"
                # Reset states for the demo file
                st.session_state.opening_remarks_topics = None
                st.session_state.qa_session_topics = None
                st.session_state.opening_remarks_summaries = {}
                st.session_state.qa_session_summaries = {}
                st.session_state.chat_history = []
                st.rerun()
        else:
            st.error("Demo file 'Q2FY24_LaurusLabs_EarningsCallTranscript.pdf' not found.")

    st.markdown("---")

    # --- Step 3: Navigation Buttons ---
    if st.session_state.data is not None:
        st.header("Navigation")
        if st.button("Opening Remarks", use_container_width=True):
            st.session_state.current_view = "Opening Remarks"
        if st.button("Q&A Session", use_container_width=True):
            st.session_state.current_view = "Q&A Session"
        if st.button("AI Assistant", use_container_width=True):
            st.session_state.current_view = "AI Assistant"

# --- Main Page Content Rendering ---
st.title("Earnings Call Analyzer")

if st.session_state.data:
    metadata = st.session_state.data["metadata"]
    st.success(f"Successfully loaded transcript for **{metadata.company_name}** on **{metadata.call_date}**")

    if st.session_state.current_view == "Opening Remarks":
        render_opening_remarks_page()
    elif st.session_state.current_view == "Q&A Session":
        render_qa_session_page()
    elif st.session_state.current_view == "AI Assistant":
        render_ai_assistant_page()
    else:
        render_welcome_page()
else:
    render_welcome_page()