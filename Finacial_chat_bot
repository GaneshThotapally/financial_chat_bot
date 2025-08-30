import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
import torch
import PyPDF2
from PIL import Image
import pytesseract  # For OCR on images
import datetime  # For timestamping chats
import threading  # For background generation
import plotly.express as px  # For charts
import pandas as pd  # For data handling

# App title with custom styling (dark blue theme inspired by the dashboard image)
st.markdown("""
    <style>
    .stApp {
        background-color: #1a1a3d;
        color: #ffffff;
    }
    .stSidebar {
        background-color: #0f0f2a;
    }
    .stButton > button {
        background-color: #6a5acd;
        color: white;
    }
    .stSelectbox {
        background-color: #2a2a5a;
        color: #ffffff;
    }
    .stChatMessage {
        background-color: #2a2a5a;
    }
    .prompt-card {
        background-color: #ff6b6b;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
    }
    .sidebar .stButton > button {
        width: 100%;
    }
    .stSpinner {
        color: #ffffff;
    }
    .metric-card {
        background-color: #2a2a5a;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
st.title("Personal Finance Dashboard")

# Sidebar with icons (mimicking the image: menu, grid, chart, search, calendar, settings)
st.sidebar.title("Menu")
st.sidebar.button("🏠 Home")
st.sidebar.button("📊 Dashboard")
st.sidebar.button("📈 Charts")
st.sidebar.button("🔍 Search")
st.sidebar.button("📅 Calendar")
st.sidebar.button("⚙️ Settings")

# Model selection in sidebar
model_options = {
    
    "Granite 7B (Better Quality)": "ibm-granite/granite-7b-instruct"
}
selected_model = st.sidebar.selectbox("Select Model:", list(model_options.keys()), index=0)
model_id = model_options[selected_model]

# Load the selected Granite model
@st.cache_resource
def load_model(model_id):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_model(model_id)

# Function to generate response with streaming
def generate_response(user_input, placeholder):
    tone = "simple, encouraging, and easy-to-understand" if st.session_state.current_chat['user_type'] == "student" else "formal, detailed, and professional"
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.current_chat['messages'][-5:]])
    uploaded = st.session_state.current_chat['uploaded_content']

    prompt = f"""
You are an advanced chartered accountant and personal finance chatbot. Respond ONLY with direct, helpful advice on the user's query. Do NOT include any user tags like 'user1', 'user2', 'assessor', or any other prefixes. Do NOT generate imaginary conversations or multiple users. Start your response immediately with the advice, in the specified tone.

For budget summaries, provide a full, detailed breakdown: calculate total expenses accurately, savings (income minus expenses if income provided, else note it's missing), percentages for each category (value/total*100, rounded to 2 decimals), and 2-3 specific suggestions to optimize spending. Use bullet points or numbered lists for clarity. If income not provided, ask for it politely at the end.

For tax operations (basic to pro): Use Indian tax slabs (new regime AY 2025-26):
- Up to ₹3,00,000: 0%
- ₹3,00,001 - ₹7,00,000: 5%
- ₹7,00,001 - ₹10,00,000: 10%
- ₹10,00,001 - ₹12,00,000: 15%
- ₹12,00,001 - ₹15,00,000: 20%
- Above ₹15,00,000: 30%
Include deductions (e.g., 80C up to ₹1.5L, 80D health insurance), exemptions, effective tax rate, TDS, advance tax, etc. Calculate precisely: total tax = sum of slab-wise tax, minus deductions/rebates (e.g., ₹17,500 rebate u/s 87A if income ≤₹7L). Use step-by-step math.

Always complete the full response without cutting off. Ensure calculations are error-free.

Tone: {tone} for {st.session_state.current_chat['user_type']}.

Query: {user_input}

Context: {context}

Uploaded: {uploaded[:2000]}

Provide complete, neat, clear response directly like a professional AI.
"""

    inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], return_tensors="pt", add_generation_prompt=True).to("cuda")
    
    # Streamer for real-time output
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = {
        "inputs": inputs,
        "max_new_tokens": 800,
        "do_sample": False,  # Greedy for accuracy
        "temperature": 0.1,
        "top_p": 1.0,
        "no_repeat_ngram_size": 5,
        "repetition_penalty": 1.3,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer
    }
    
    # Run generation in background thread to avoid blocking UI
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream the response to placeholder
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        placeholder.markdown(generated_text)
    
    # Post-process if needed
    if generated_text.lower().startswith(("user", "assessor", "assistant:")):
        generated_text = generated_text.split(":", 1)[-1].strip() if ":" in generated_text else generated_text
    
    return generated_text

# Session state for chat history (multiple threads)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of past chats: [{'title': str, 'messages': list, 'user_type': str, 'uploaded_content': str}]

if "current_chat" not in st.session_state:
    st.session_state.current_chat = {
        'title': None,
        'messages': [{"role": "assistant", "content": "Hello! Select your demographic and ask a question."}],
        'user_type': "student",
        'uploaded_content': ""
    }

# Sidebar for New Chat and History
st.sidebar.title("Chats")
if st.sidebar.button("New Chat"):
    if st.session_state.current_chat['messages'] and len(st.session_state.current_chat['messages']) > 1:
        # Auto-title based on first user message or timestamp
        first_user_msg = next((msg['content'] for msg in st.session_state.current_chat['messages'] if msg['role'] == 'user'), "Untitled Chat")
        title = first_user_msg[:50] + "..." if len(first_user_msg) > 50 else first_user_msg
        st.session_state.current_chat['title'] = title or datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state.chat_history.append(st.session_state.current_chat)
    # Start new chat
    st.session_state.current_chat = {
        'title': None,
        'messages': [{"role": "assistant", "content": "Hello! Select your demographic and ask a question."}],
        'user_type': "student",
        'uploaded_content': ""
    }

# Display chat history in sidebar
for idx, chat in enumerate(st.session_state.chat_history):
    if st.sidebar.button(chat['title'], key=f"hist_{idx}"):
        # Load selected chat
        st.session_state.current_chat = chat
        st.session_state.chat_history.pop(idx)  # Optional: Remove from history or keep

# Main dashboard layout with columns (mimicking the image)
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("AI Assistant")
    # Blue orb placeholder (use emoji or image)
    st.image("https://via.placeholder.com/200x200/0000FF/808080?text=AI+Orb", use_container_width=True)
    st.button("Try Now")
    st.write("Analyze your finances, budgets, taxes, and more. Ask me anything!")

with col_right:
    # Total Savings metric (dummy data)
    st.metric(label="Total Savings", value="₹90,744", delta="5%")

    # Pie chart for expense categories (dummy data)
    data_pie = pd.DataFrame({
        'Category': ['Rent', 'Groceries', 'Eating Out', 'Transport'],
        'Amount': [20000, 8000, 5000, 4000]
    })
    fig_pie = px.pie(data_pie, values='Amount', names='Category', hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_pie, use_container_width=True)

    # Bar chart for monthly expenses (dummy)
    data_bar = pd.DataFrame({
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'Expenses': [50000, 45000, 48000, 52000, 47000, 49000]
    })
    fig_bar = px.bar(data_bar, x='Month', y='Expenses', color_discrete_sequence=['#6a5acd'])
    st.plotly_chart(fig_bar, use_container_width=True)

# Gauge and Line chart below (full width)
col_gauge, col_line = st.columns(2)

with col_gauge:
    # Gauge for Credit Score (dummy)
    fig_gauge = px.pie(values=[80, 20], names=['Used', 'Available'], hole=0.5, color_discrete_sequence=['#6a5acd', '#d3d3d3'])
    fig_gauge.update_traces(textinfo='none')
    fig_gauge.add_annotation(text='803', x=0.5, y=0.5, showarrow=False, font_size=40)
    fig_gauge.update_layout(title="Credit Rate")
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_line:
    # Revenue Trend line chart (dummy)
    data_line = pd.DataFrame({
        'Date': pd.date_range(start='2024-01-01', periods=6, freq='M'),
        'Revenue': [1.2, 1.5, 1.3, 1.8, 2.0, 1.7]
    })
    fig_line = px.line(data_line, x='Date', y='Revenue', markers=True, color_discrete_sequence=['#00bfff'])
    st.plotly_chart(fig_line, use_container_width=True)

# Select user type below charts
user_type = st.selectbox("Select your demographic:", ["Student", "Professional"])
st.session_state.current_chat['user_type'] = user_type.lower()

# Display chat history
for message in st.session_state.current_chat['messages']:
    with st.chat_message(message["role"], avatar="🧑‍💼" if message["role"] == "assistant" else "👤"):
        st.markdown(message["content"])

# Pre-loaded prompts if no user messages yet
if len(st.session_state.current_chat['messages']) == 1:  # Only initial assistant message
    st.subheader("Start Conversation")
    prompts = [
        "How can I save money as a student?",
        "Generate a budget summary for ₹80,000 income.",
        "Tax tips for professionals.",
        "Investment suggestions for beginners.",
        "Analyze my spending: ₹20,000 rent, ₹8,000 groceries."
    ]
    cols = st.columns(3)  # Display in grid like the image
    for i, prompt in enumerate(prompts):
        with cols[i % 3]:
            if st.button(prompt, key=f"prompt_{i}"):
                # Simulate user input
                st.session_state.current_chat['messages'].append({"role": "user", "content": prompt})
                with st.chat_message("assistant", avatar="🧑‍💼"):
                    placeholder = st.empty()
                    response = generate_response(prompt, placeholder)
                st.session_state.current_chat['messages'].append({"role": "assistant", "content": response})
                st.rerun()  # Refresh if needed

# Input bar with upload integrated at bottom
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.chat_input("Ask about savings, taxes, investments, budget, or spending:")
with col2:
    uploaded_file = st.file_uploader("", type=["pdf", "jpg", "png", "jpeg"], label_visibility="collapsed")

# Process upload
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        st.session_state.current_chat['uploaded_content'] = text
        st.sidebar.success("PDF uploaded!")
    elif uploaded_file.type.startswith("image/"):
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)
        st.session_state.current_chat['uploaded_content'] = text
        st.sidebar.success("Image uploaded!")

# Generate response if input
if user_input:
    st.session_state.current_chat['messages'].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    with st.chat_message("assistant", avatar="🧑‍💼"):
        placeholder = st.empty()
        with st.spinner("Thinking..."):
            response = generate_response(user_input, placeholder)
    st.session_state.current_chat['messages'].append({"role": "assistant", "content": response})
