"""Enhanced Streamlit interface for Analysis Assistant."""
import streamlit as st
import sys
from pathlib import Path
from io import StringIO
import contextlib

# Add src to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

import src.config as config
from src.agents.ollama_client import test_ollama_connection
from src.agents.openai_client import test_openai_connection
from src.agents.analysis_assistant import AnalysisAssistant
from src.data_dictionary import DataDictionary

# Page configuration
st.set_page_config(
    page_title="Taylor Swift Analysis Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Bejeweled Bot - Subtle & Elegant
st.markdown("""
    <style>
    /* Sparkle animations */
    @keyframes sparkle {
        0%, 100% { opacity: 0; transform: scale(0); }
        50% { opacity: 1; transform: scale(1); }
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Main background with subtle gradient */
    .main {
        background: linear-gradient(135deg, #faf8f3 0%, #f5f0fa 50%, #faf8f3 100%);
    }

    /* Sparkle decorations */
    .sparkle {
        display: inline-block;
        animation: sparkle 2s ease-in-out infinite;
        font-size: 1.2rem;
    }

    .sparkle:nth-child(2) { animation-delay: 0.3s; }
    .sparkle:nth-child(3) { animation-delay: 0.6s; }
    .sparkle:nth-child(4) { animation-delay: 0.9s; }

    /* Elegant button styling */
    .stButton>button {
        background: linear-gradient(135deg, #9b7ba6 0%, #8b5a8e 100%);
        color: white;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        border: none;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(139, 90, 142, 0.2);
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #8b5a8e 0%, #7b4a7e 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(139, 90, 142, 0.35);
    }

    /* Sidebar with soft elegance */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f9f5f0 0%, #f4e8d8 100%);
    }

    /* Title with vibrant shimmer */
    h1 {
        background: linear-gradient(90deg, 
            #ff69b4 0%,    /* Hot pink */
            #ff1493 20%,   /* Deep pink */
            #9370db 40%,   /* Medium purple */
            #4169e1 60%,   /* Royal blue */
            #9370db 80%,   /* Medium purple */
            #ff69b4 100%   /* Hot pink */
        );
        background-size: 200% auto;
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 4s linear infinite;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 2px 6px rgba(255, 105, 180, 0.4));
    }

    h2, h3 {
        color: #a67c94;
    }

    .subtitle {
        text-align: center;
        color: #6b4a6e;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-style: italic;
        opacity: 0.85;
    }

    /* Glassmorphism chat messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(139, 90, 142, 0.15);
        box-shadow: 0 4px 12px rgba(139, 90, 142, 0.08);
        animation: fadeIn 0.4s ease-out;
    }

    /* Elegant input styling */
    .stChatInput {
        border-radius: 20px;
        border: 2px solid rgba(139, 90, 142, 0.2);
        transition: border-color 0.3s ease;
    }

    .stChatInput:focus-within {
        border-color: rgba(139, 90, 142, 0.5);
        box-shadow: 0 0 0 3px rgba(139, 90, 142, 0.1);
    }

    /* Status badges with soft glow */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 500;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    }

    .status-connected {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        color: #2e7d32;
    }

    .status-disconnected {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        color: #c62828;
    }

    /* Info card elegance */
    .info-card {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(8px);
        border-left: 4px solid #8b5a8e;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(139, 90, 142, 0.1);
    }

    /* Console output - elegant dark theme */
    .console-output {
        background: linear-gradient(135deg, #2a2a2a 0%, #1e1e1e 100%);
        color: #d4d4d4;
        font-family: 'SF Mono', 'Monaco', 'Courier New', monospace;
        font-size: 0.85rem;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid rgba(139, 90, 142, 0.2);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    .console-line {
        margin: 0.2rem 0;
        line-height: 1.5;
    }

    .console-header {
        color: #b399b8;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        letter-spacing: 0.5px;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(139, 90, 142, 0.05);
        border-radius: 8px;
        transition: background 0.3s ease;
    }

    .streamlit-expanderHeader:hover {
        background: rgba(139, 90, 142, 0.1);
    }

    /* Smooth scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(139, 90, 142, 0.05);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(139, 90, 142, 0.3);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(139, 90, 142, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

@contextlib.contextmanager
def capture_stdout():
    """Context manager to capture stdout."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield sys.stdout
    finally:
        sys.stdout = old_stdout

def format_console_output(output: str) -> str:
    """Format console output for display."""
    if not output.strip():
        return ""
    
    lines = output.strip().split('\n')
    formatted = '<div class="console-output">'
    formatted += '<div class="console-header">System Output:</div>'
    
    for line in lines:
        # Add syntax highlighting for different types of messages
        if '[Agent requesting data' in line or '[FILTER]' in line or '[AGGREGATION]' in line:
            formatted += f'<div class="console-line" style="color: #4ec9b0;">{line}</div>'
        elif '[WARNING]' in line or 'Warning' in line:
            formatted += f'<div class="console-line" style="color: #ffa500;">{line}</div>'
        elif 'ERROR' in line or 'Error' in line:
            formatted += f'<div class="console-line" style="color: #f48771;">{line}</div>'
        elif 'Loaded' in line or 'initialized' in line:
            formatted += f'<div class="console-line" style="color: #89d185;">{line}</div>'
        else:
            formatted += f'<div class="console-line">{line}</div>'
    
    formatted += '</div>'
    return formatted

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'connection_tested' not in st.session_state:
    st.session_state.connection_tested = False
if 'connection_status' not in st.session_state:
    st.session_state.connection_status = None
if 'agent_initialized' not in st.session_state:
    st.session_state.agent_initialized = False
if 'show_console' not in st.session_state:
    st.session_state.show_console = True

# Sidebar
with st.sidebar:
    st.markdown("## Connection & Settings")
    
    # Connection status
    st.markdown("### LLM Connection")
    
    if st.button("Test Connection", use_container_width=True):
        with st.spinner("Testing connection..."):
            if not config.USE_OPENAI:
                st.session_state.connection_status = test_ollama_connection()
            else:
                st.session_state.connection_status = test_openai_connection()
            st.session_state.connection_tested = True
    
    if st.session_state.connection_tested:
        if st.session_state.connection_status:
            st.markdown('<span class="status-badge status-connected">Connected</span>', unsafe_allow_html=True)
            st.success(f"Using model: {config.MODEL}")
        else:
            st.markdown('<span class="status-badge status-disconnected">Disconnected</span>', unsafe_allow_html=True)
            st.error("Connection failed. Please check your LLM service.")
    
    st.markdown("---")
    
    # Initialize agent
    st.markdown("### Agent Status")
    
    if not st.session_state.agent_initialized:
        if st.button("Initialize Agent", use_container_width=True, type="primary"):
            with st.spinner("Initializing Analysis Assistant..."):
                try:
                    # Capture initialization output
                    with capture_stdout() as output:
                        st.session_state.agent = AnalysisAssistant()
                    
                    init_output = output.getvalue()
                    if init_output:
                        st.markdown(format_console_output(init_output), unsafe_allow_html=True)
                    
                    st.session_state.agent_initialized = True
                    st.success("Agent ready!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to initialize agent: {str(e)}")
    else:
        st.markdown('<span class="status-badge status-connected">Agent Ready</span>', unsafe_allow_html=True)
        
        if st.button("Reset Agent", use_container_width=True):
            if st.session_state.agent:
                st.session_state.agent.reset()
            st.session_state.messages = []
            st.success("Agent and conversation reset!")
            st.rerun()
    
    st.markdown("---")
    
    # Console output toggle
    st.markdown("### Display Options")
    st.session_state.show_console = st.checkbox(
        "Show System Output", 
        value=st.session_state.show_console,
        help="Display internal processing information"
    )
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("### Quick Actions")

    #col1, col2 = st.columns(2)

    #with col1:
    if st.button("Get AI Insights", use_container_width=True):
        if st.session_state.agent_initialized:
            st.session_state.insight_trigger = True
            st.rerun()

    #with col2:
    if st.button("Sample Questions", use_container_width=True):
        if st.session_state.agent_initialized:
            st.session_state.questions_trigger = True
            st.rerun()

    if st.button("View Data Columns", use_container_width=True):
        st.session_state.show_columns = True
        st.rerun()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    
    # Help section
    with st.expander("Help & Tips"):
        st.markdown("""
        **How to use:**
        - Ask complex questions to trigger Chain-of-Thought reasoning
        - Request AI-generated insights for research ideas
        - View available data columns for context
        - Reset agent to clear conversation history
        
        **Example questions:**
        - "What are the lyrical themes across eras?"
        - "Which songs have the highest danceability?"
        - "Compare folklore and evermore albums"
        - "What makes reputation era unique?"
        """)
    
    # About section
    with st.expander("About"):
        st.markdown("""
        **Enhanced Analysis Assistant**
        
        Features:
        - Chain-of-Thought reasoning
        - Dynamic data retrieval
        - Context-aware responses
        - Conversation memory
        
        Built with Streamlit and powered by LLMs
        """)

#<span class="sparkle" style="color: #ff69b4;">✨</span>
#<span class="sparkle" style="color: #9370db;">💎</span>
#<span class="sparkle" style="color: #4169e1;">💎</span>
#<span class="sparkle" style="color: #ff1493;">✨</span>

# Main content with colorful sparkles
st.markdown("""
    <h1>
        Bejeweled Bot
    </h1>
""", unsafe_allow_html=True)

st.markdown(
    """
    <h2 style='text-align: center; 
               background: linear-gradient(90deg, #ff69b4 0%, #9370db 50%, #4169e1 100%);
               background-clip: text;
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               margin-top: -30px; 
               font-weight: 600;
               filter: drop-shadow(0 2px 4px rgba(255, 105, 180, 0.3));'>
        Because data never goes out of style
    </h2>
    """,
    unsafe_allow_html=True
)

#<span class="sparkle" style="color: #ff1493;">✨</span>
#<span class="sparkle" style="color: #9370db;">✨</span>

st.markdown(
    """
    <style>
    .subtitle {
        font-style: italic;  /* Make the text italic */
    }
    .subtitle .sparkle {
        font-style: normal;  /* Override italics for the emojis */
        display: inline-block;
        transform: none;     /* Prevent skew or inherited transforms */
    }
    </style>

    <p class="subtitle">
        <span class="sparkle" style="color: #ff69b4;">💎</span>  
        Chain-of-Thought Reasoning + Dynamic Data Retrieval 
        <span class="sparkle" style="color: #4169e1;">💎</span>
    </p>
    """,
    unsafe_allow_html=True
)

# Show columns modal
if 'show_columns' in st.session_state and st.session_state.show_columns:
    with st.expander("Available Data Columns", expanded=True):
        cols = st.columns(2)
        col_items = list(DataDictionary.COLUMNS.items())
        mid = len(col_items) // 2
        
        with cols[0]:
            for col, desc in col_items[:mid]:
                st.markdown(f"**{col}**")
                st.markdown(f"<small>{desc}</small>", unsafe_allow_html=True)
                st.markdown("")
        
        with cols[1]:
            for col, desc in col_items[mid:]:
                st.markdown(f"**{col}**")
                st.markdown(f"<small>{desc}</small>", unsafe_allow_html=True)
                st.markdown("")
        
        if st.button("Close"):
            st.session_state.show_columns = False
            st.rerun()

# Handle insight trigger
if 'insight_trigger' in st.session_state and st.session_state.insight_trigger:
    if st.session_state.agent_initialized and st.session_state.agent:
        with st.spinner("Generating AI insights..."):
            try:
                # Capture console output during insight generation
                with capture_stdout() as output:
                    response = st.session_state.agent.suggest_insights()
                
                console_output = output.getvalue()
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "type": "insights",
                    "console": console_output if console_output else None
                })
                st.session_state.insight_trigger = False
                st.rerun()
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")
                st.session_state.insight_trigger = False

# Handle questions trigger
if 'questions_trigger' in st.session_state and st.session_state.questions_trigger:
    if st.session_state.agent_initialized and st.session_state.agent:
        with st.spinner("Generating sample questions..."):
            try:
                # Capture console output during question generation
                with capture_stdout() as output:
                    response = st.session_state.agent.suggest_questions()

                console_output = output.getvalue()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "type": "questions",
                    "console": console_output if console_output else None
                })
                st.session_state.questions_trigger = False
                st.rerun()
            except Exception as e:
                st.error(f"Error generating questions: {str(e)}")
                st.session_state.questions_trigger = False

# Chat interface
if not st.session_state.agent_initialized:
    st.info("Please initialize the agent using the sidebar to start chatting.")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show console output if available and enabled
            if st.session_state.show_console and message.get("console"):
                st.markdown(format_console_output(message["console"]), unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about Taylor Swift's music..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            # Create placeholders for response and console output
            response_placeholder = st.empty()
            console_placeholder = st.empty()
            
            with st.spinner("Thinking..."):
                try:
                    # Capture console output during processing
                    with capture_stdout() as output:
                        response = st.session_state.agent.ask(prompt)
                    
                    console_output = output.getvalue()
                    
                    # Display response
                    response_placeholder.markdown(response)
                    
                    # Display console output if enabled
                    if st.session_state.show_console and console_output:
                        console_placeholder.markdown(
                            format_console_output(console_output), 
                            unsafe_allow_html=True
                        )
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "type": "response",
                        "console": console_output if console_output else None
                    })
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    response_placeholder.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "type": "error"
                    })

# Footer with colorful elegance
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #8b5a8e; padding: 1.5rem;'>
    <p style='font-size: 0.95rem; margin-bottom: 0.3rem;'>
        <span class="sparkle" style="color: #ff1493;">✨</span> 
        <span style='background: linear-gradient(90deg, #ff69b4, #9370db, #4169e1);
                     background-clip: text;
                     -webkit-background-clip: text;
                     -webkit-text-fill-color: transparent;
                     font-weight: 600;'>
            Powered by Chain-of-Thought Reasoning
        </span>
        <span class="sparkle" style="color: #9370db;">✨</span>
    </p>
    <p style='font-size: 0.85rem; color: #a67c94; opacity: 0.9;'>
        Analyzing Taylor Swift's discography with advanced AI
    </p>
</div>
""", unsafe_allow_html=True)
