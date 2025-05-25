import os
import asyncio
import platform
import streamlit as st
from dotenv import load_dotenv
import chatbot

# Fix for Windows event loop
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

def initialize_session_state():
    required_keys = {
        "conversation": None,
        "chat_history": [],
        "document_processed": False,
        "user_info": {},
        "appointment_info": {},
        "current_form": None,
        "agent": None,
        "vector_store": None,
        "recursion_depth": 0,
    }
    
    for key, default_value in required_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def handle_user_info_form():
    with st.form("user_info"):
        st.subheader("Contact Information")
        name = st.text_input("Full Name")
        phone = st.text_input("Phone Number (+countrycode)")
        email = st.text_input("Email")
        
        if st.form_submit_button("Submit"):
            try:
                user_info = chatbot.UserInfo(name=name, phone=phone, email=email)
                st.session_state.user_info = user_info.model_dump()
                st.session_state.current_form = None
                st.rerun()
            except Exception as e:
                st.error(str(e))

def handle_appointment_form():
    with st.form("appointment"):
        st.subheader("Schedule Appointment")
        date_str = st.text_input("Date/Time Description")
        purpose = st.text_area("Appointment Purpose")
        
        if st.form_submit_button("Schedule"):
            try:
                appointment = chatbot.AppointmentInfo(
                    user_info=chatbot.UserInfo(**st.session_state.user_info),
                    date=date_str,
                    purpose=purpose
                )
                st.session_state.appointment_info = appointment.model_dump()
                st.session_state.current_form = None
                st.rerun()
            except Exception as e:
                st.error(str(e))

def main():
    st.set_page_config(
        page_title="Professional Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    initialize_session_state()
    
    st.title("Professional Chatbot System")
    st.markdown("AI-powered assistant for document Q&A and appointment management")
    
    with st.sidebar:
        st.header("Document Management")
        uploaded_files = st.file_uploader(
            "Upload PDF/TXT files",
            type=chatbot.ALLOWED_FILE_TYPES,
            accept_multiple_files=True
        )
        if st.button("Process Documents"):
            with st.spinner("Processing..."):
                success, message, conversation, vector_store = chatbot.process_documents(uploaded_files)
                if success:
                    st.session_state.conversation = conversation
                    st.session_state.vector_store = vector_store
                    st.session_state.document_processed = True
                    st.success(message)
                else:
                    st.error(message)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Conversation")
        
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])
        
        if st.session_state.current_form == "user_info":
            handle_user_info_form()
        elif st.session_state.current_form == "appointment":
            handle_appointment_form()
        
        if prompt := st.chat_input("Your message..."):
            st.chat_message("user").write(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            try:
                if not st.session_state.agent:
                    st.session_state.agent = chatbot.initialize_agent()
                
                response = st.session_state.agent.invoke({
                    "messages": [HumanMessage(content=prompt)],
                    "recursion_depth": 0
                })
                reply = response["messages"][-1].content
                
                st.chat_message("assistant").write(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
            except Exception as e:
                error_msg = f"System error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
    
    with col2:
        st.subheader("System Status")
        if st.session_state.document_processed:
            st.success("✅ Documents Ready")
        else:
            st.warning("⚠️ No Documents")
        
        if st.session_state.user_info:
            st.info("📝 User Info Collected")
        
        if st.session_state.appointment_info:
            st.success("📅 Appointment Scheduled")
        
        st.markdown("---")
        st.markdown("**Capabilities**")
        st.markdown("- Document Q&A\n- Appointment Booking\n- Callback Requests")

if __name__ == "__main__":
    os.makedirs("./temp", exist_ok=True)
    main()