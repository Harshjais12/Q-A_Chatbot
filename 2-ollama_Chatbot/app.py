import streamlit as st
import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

# Load environment variables from .env file (for local development)
load_dotenv()

# Set LangChain environment variables for tracing (optional, but good practice)
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true" # Corrected typo: "TRACHING" to "TRACING"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot With OLLAMA"

# Define the chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question:{question}")
    ]
)

def generate_response(question, llm_model_name, temperature, max_tokens):
    """
    Generates a response from the Ollama model.

    Args:
        question (str): The user's query.
        llm_model_name (str): The name of the Ollama model to use (e.g., "llama3.2:1b").
        temperature (float): The creativity of the model's response (0.0 to 1.0).
        max_tokens (int): The maximum number of tokens in the model's response.
    
    Returns:
        str: The generated answer from the LLM.
    """
    # Get the Ollama base URL from Streamlit secrets.
    # For local development, it defaults to 'http://localhost:11434'.
    # When deploying, you MUST set OLLAMA_BASE_URL in your Streamlit Cloud secrets
    # to the actual public URL/IP of your Ollama server.
    ollama_base_url = st.secrets.get("OLLAMA_BASE_URL", "http://localhost:11434")

    # Initialize the Ollama model with specified parameters
    llm = Ollama(
        model=llm_model_name,
        temperature=temperature,
        num_ctx=max_tokens, # num_ctx controls the context window size (similar to max_tokens for input/output)
        base_url=ollama_base_url # Pass the base URL to connect to the Ollama server
    )

    # Define the output parser
    output_parser = StrOutputParser()

    # Create the LangChain processing chain: prompt -> llm -> output_parser
    chain = prompt | llm | output_parser

    # Invoke the chain with the user's question
    answer = chain.invoke({'question': question})
    return answer

# --- Streamlit UI ---
st.title("Enhanced Q&A Chatbot With Ollama")

st.sidebar.title("Settings")

# Select Ollama model (ensure this model is available on your Ollama server)
llm_model = st.sidebar.selectbox("Select an Ollama Model", ["llama3.2:1b"])

# Temperature slider for model creativity
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)

# Max Tokens slider for response length
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

# Generate and display response if user provides input
if user_input:
    # Pass all necessary parameters to the generate_response function
    response = generate_response(user_input, llm_model, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")
