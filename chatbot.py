import os
import re
import phonenumbers
from datetime import datetime
import dateparser
from typing import List, TypedDict, Annotated
from pydantic import BaseModel, field_validator, EmailStr
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import END, StateGraph
from langchain_core.tools import tool
import streamlit as st

# Constants
ALLOWED_FILE_TYPES = ["pdf", "txt"]
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_REPO_ID = "google/flan-t5-xxl"

class UserInfo(BaseModel):
    name: str
    phone: str
    email: EmailStr
    
    @field_validator('phone')
    def validate_phone(cls, v):
        try:
            parsed = phonenumbers.parse(v, None)
            if not phonenumbers.is_valid_number(parsed):
                raise ValueError("Invalid phone number")
            return v
        except Exception as e:
            raise ValueError("Phone number must be in international format (e.g., +12125552368)")

class AppointmentInfo(BaseModel):
    user_info: UserInfo
    date: str
    purpose: str
    
    @field_validator('date')
    def validate_date(cls, v):
        parsed_date = dateparser.parse(v)
        if not parsed_date:
            raise ValueError("Could not understand the date. Please try again with a different format.")
        return parsed_date.strftime("%Y-%m-%d %H:%M:%S")

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "messages"]
    recursion_depth: Annotated[int, "recursion_depth"]

@tool
def document_qa(query: str) -> str:
    """Answer questions from uploaded documents."""
    try:
        if not st.session_state.get("vector_store"):
            return "Please upload and process documents first."
        
        if not st.session_state.conversation:
            return "Document processing incomplete. Please re-process documents."
        
        result = st.session_state.conversation({"question": query})
        
        if not result or "answer" not in result:
            return "No relevant information found in documents."
        
        if result.get("source_documents"):
            return f"{result['answer']}\n\nSources: {len(result['source_documents'])} documents"
        return result["answer"]
    
    except Exception as e:
        return f"Document query error: {str(e)}"

@tool
def appointment_booking(date_str: str, purpose: str) -> str:
    """Book an appointment by parsing natural language date."""
    try:
        parsed_date = dateparser.parse(date_str)
        if not parsed_date:
            return "Could not understand the date. Please try again."
        
        if not st.session_state.get("user_info"):
            st.session_state.current_form = "user_info"
            return "Please provide contact information first."
        
        st.session_state.appointment_info = {
            "date": parsed_date.strftime("%Y-%m-%d %H:%M:%S"),
            "purpose": purpose
        }
        st.session_state.current_form = None
        return f"Appointment booked for {parsed_date.strftime('%Y-%m-%d %H:%M')}"
    except Exception as e:
        return f"Error booking appointment: {str(e)}"

@tool
def callback_request(reason: str) -> str:
    """Request a callback from support team."""
    if not st.session_state.get("user_info"):
        st.session_state.current_form = "user_info"
        return "Please provide contact information first."
    
    return f"Callback requested: {reason}. We'll contact you at {st.session_state.user_info['phone']}."

@tool
def get_current_date() -> str:
    """Get current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def process_documents(uploaded_files):
    """Process uploaded documents and return status, message, conversation chain, and vector store."""
    try:
        if not uploaded_files:
            return (False, "No files uploaded", None, None)
        
        documents = []
        for file in uploaded_files:
            try:
                file_extension = os.path.splitext(file.name)[1][1:].lower()
                if file_extension not in ALLOWED_FILE_TYPES:
                    continue
                
                temp_file_path = f"./temp/{file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(file.getbuffer())
                
                try:
                    loader = PyPDFLoader(temp_file_path) if file_extension == "pdf" else TextLoader(temp_file_path)
                    loaded_docs = loader.load()
                    documents.extend(loaded_docs)
                except Exception as e:
                    return (False, f"Error loading {file.name}: {str(e)}", None, None)
                finally:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
            except Exception as e:
                return (False, f"Error processing {file.name}: {str(e)}", None, None)
        
        if not documents:
            return (False, "No valid documents found", None, None)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        if vector_store.index.ntotal == 0:
            return (False, "No documents could be processed into vectors", None, None)
        
        llm = HuggingFaceEndpoint(
            repo_id=LLM_REPO_ID,
            temperature=0.5,
            max_new_tokens=512
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True
        )
        
        return (True, "Documents processed successfully", conversation, vector_store)
    
    except Exception as e:
        return (False, f"Document processing failed: {str(e)}", None, None)

def initialize_agent():
    tools = [document_qa, appointment_booking, callback_request, get_current_date, DuckDuckGoSearchRun()]
    
    llm = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        temperature=0.3,
        max_new_tokens=512
    )
    
    def agent(state: AgentState):
        if state["recursion_depth"] >= 5:
            return {"messages": [AIMessage(content="Let's start fresh. How can I help you?")], "recursion_depth": 0}
        
        try:
            response = llm.invoke(state["messages"])
            return {
                "messages": [AIMessage(content=response)],
                "recursion_depth": state["recursion_depth"] + 1
            }
        except Exception:
            return {"messages": [AIMessage(content="Let's try again.")], "recursion_depth": 0}

    def tools_node(state: AgentState):
        if state["recursion_depth"] >= 5:
            return {"messages": [AIMessage(content="Please ask a new question.")], "recursion_depth": 0}
        
        last_message = state["messages"][-1].content
        for t in tools:
            if re.search(rf'\b{t.name}\b', last_message, re.IGNORECASE):
                try:
                    args = last_message.split(t.name)[-1].strip()
                    result = t.invoke(args)
                    return {
                        "messages": [AIMessage(content=result)],
                        "recursion_depth": state["recursion_depth"] + 1
                    }
                except Exception as e:
                    return {
                        "messages": [AIMessage(content=f"Error: {str(e)}")],
                        "recursion_depth": state["recursion_depth"] + 1
                    }
        return {"messages": [AIMessage(content="How can I assist you?")], "recursion_depth": state["recursion_depth"] + 1}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tools_node)
    workflow.set_entry_point("agent")
    
    def decide_edges(state: AgentState):
        if state["recursion_depth"] >= 5:
            return END
        last_message = state["messages"][-1].content.lower()
        if any(re.search(rf'\b{t.name}\b', last_message) for t in tools):
            return "tools"
        return END
    
    workflow.add_conditional_edges("agent", decide_edges)
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()