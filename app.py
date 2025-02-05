import pandas as pd
import re
import streamlit as st
from ollama import chat

# Set Streamlit page configuration (optional)
st.set_page_config(page_title="TCET AI", layout="centered")

def format_reasoning_response(thinking_content):
    """Format assistant content by removing think tags."""
    return (
        thinking_content.replace("<think>\n\n</think>", "")
        .replace("<think>", "")
        .replace("</think>", "")
    )

def display_message(message):
    """Display a single message in the chat interface."""
    role = "user" if message["role"] == "user" else "assistant"
    with st.chat_message(role):
        if role == "assistant":
            display_assistant_message(message["content"])
        else:
            st.markdown(message["content"])

def display_assistant_message(content):
    """Display assistant message with thinking content if present."""
    pattern = r"<think>(.*?)</think>"
    think_match = re.search(pattern, content, re.DOTALL)
    if think_match:
        think_content = think_match.group(0)
        response_content = content.replace(think_content, "")
        think_content = format_reasoning_response(think_content)
        with st.expander("Thinking complete!"):
            st.markdown(think_content)
        st.markdown(response_content)
    else:
        st.markdown(content)

def display_chat_history():
    """Display all previous messages in the chat history."""
    for message in st.session_state["messages"]:
        if message["role"] != "system":  # Skip system messages
            display_message(message)

def process_thinking_phase(stream):
    """Process the thinking phase of the assistant's response."""
    thinking_content = ""
    with st.status("Thinking...", expanded=True) as status:
        think_placeholder = st.empty()
        
        for chunk in stream:
            content = chunk["message"]["content"] or ""
            thinking_content += content
            
            if "<think>" in content:
                continue
            if "</think>" in content:
                content = content.replace("</think>", "")
                status.update(label="Thinking complete!", state="complete", expanded=False)
                break
            think_placeholder.markdown(format_reasoning_response(thinking_content))
    
    return thinking_content

def process_response_phase(stream):
    """Process the response phase of the assistant's response."""
    response_placeholder = st.empty()
    response_content = ""
    for chunk in stream:
        content = chunk["message"]["content"] or ""
        response_content += content
        response_placeholder.markdown(response_content)
    return response_content

@st.cache_resource
def get_chat_model():
    """Get a cached instance of the chat model."""
    return lambda messages: chat(
        model="deepseek-r1:8b",
        messages=messages,
        stream=True,
    )

def handle_user_input():
    """Handle new user input and generate assistant response."""
    if user_input := st.chat_input("Ask about the data..."):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            chat_model = get_chat_model()
            stream = chat_model(st.session_state["messages"])
            
            thinking_content = process_thinking_phase(stream)
            response_content = process_response_phase(stream)
            
            # Save the complete response
            st.session_state["messages"].append(
                {"role": "assistant", "content": thinking_content + response_content}
            )

def main():
    """Main function to handle the chat interface and streaming responses."""
    st.markdown("""
    # TCET RAG System with Data Insights
    """, unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'>In active development by Dhruv Paste(TE ITA 66) and Aditya Patil(TE ITA 65)</h6>", unsafe_allow_html=True)
    
    # File uploader for Excel or CSV
    uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Read the file into a DataFrame based on its type
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        
        # Show the first few rows of the dataframe to the user
        st.write(df.head())
        
        # Generate insights from the data (simple summary)
        data_insights = generate_data_insights(df)
        
        # Add insights to the system message to set the context for the assistant
        st.session_state["messages"].append({"role": "system", "content": f"Here are some insights from the uploaded data: {data_insights}"})
    
    display_chat_history()
    handle_user_input()

def generate_data_insights(df):
    """Generate simple insights like column names and basic stats."""
    summary = f"The dataset has {len(df)} rows and {len(df.columns)} columns. The columns are: {', '.join(df.columns)}."
    numeric_summary = df.describe().to_string()  # Basic statistics for numerical columns
    return f"{summary} Here are some basic statistics:\n{numeric_summary}"

if __name__ == "__main__":
    # Initialize session state
    system_name = "April"
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "system", 
                "content": f"You are a helpful AI assistant for Thakur College of Engineering and Technology(TCET), and your name is {system_name}. I can help with analyzing uploaded data."
            }
        ]
    main()
