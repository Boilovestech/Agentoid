import streamlit as st
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chains import LLMMathChain, LLMChain
import os
# Page config
st.set_page_config(
    page_title="Agentoid",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS with animations and modern UI elements
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    .stApp {
        max-width: 800px;
        margin: 0 auto;
        font-family: 'Inter', sans-serif;
        background: #000000;
        color: #ffffff;
    }
    
    .stTextInput {
        margin-bottom: 20px;
    }
    
    .title {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeIn 1s ease-in;
    }
    
    .subtitle {
        color: #ffffff;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        transition: transform 0.2s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .response-container {
        background: #1a1a1a;
        color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        margin: 20px 0;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideIn {
        from {
            transform: translateY(20px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    .stMarkdown {
        animation: fadeIn 1s ease-in;
        color: #ffffff;
    }

    .stTextInput>div>div>input {
        color: #ffffff;
        background-color: #1a1a1a;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description with custom class
st.markdown("<h1 class='title'>🤖 Agentoid</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Your AI-powered assistant for solving math, ai enhanced searching and more!</div>", unsafe_allow_html=True)
st.markdown("---")

load_dotenv()

# Initialize components in a function to avoid rerunning on every interaction
@st.cache_resource
def initialize_components():
    try:
        model = ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=1,
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        wikipedia = WikipediaAPIWrapper()
        problem_chain = LLMMathChain.from_llm(llm=model)
        
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful assistant.Be more straight to point especially when it comes to math and just give no explanation and just the answer. Give a short and concise answer.Please cite the sources with links its mandotary to have links with the source names.Dont genrate an answer without using the tools and using the info provided by he tools. If you don't know the answer, just say that you don't know.",
            ),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        tools = [
            Tool(
                name="DuckDuckGo",
                func=DuckDuckGoSearchAPIWrapper().run,
                description="useful for when you need to answer questions about current events.",
            ),
            Tool(
                name="Wikipedia",
                func=wikipedia.run,
                description="useful for when you need to get factual info even if a bit outdated somethimes.",
            ),
            Tool(
                name="Math solver",
                func=problem_chain.run,
                description="Useful for when you need to answer questions about math like math questions. This tool is only for math questions and nothing else. Only input math expressions."
            )
        ]
        
        agent = create_tool_calling_agent(llm=model, tools=tools, prompt=prompt)
        executor = AgentExecutor(tools=tools, agent=agent, verbose=True)
        
        return executor
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        return None

# Initialize the agent executor
executor = initialize_components()

# Create the chat interface with modern styling
st.markdown("<h3 class='title'>Ask me anything!</h3>", unsafe_allow_html=True)
user_input = st.text_input("Your question:", placeholder="Type your question here...", key="user_input")

# Add a submit button
if st.button("Send", type="primary"):
    if user_input:
        with st.spinner("Thinking..."):
            try:
                # Create a container for the response with custom styling
                response_container = st.container()
                with response_container:
                    st.markdown("<div class='response-container'>", unsafe_allow_html=True)
                    st.markdown("<h3>Response:</h3>", unsafe_allow_html=True)
                    response = executor.invoke({"input": user_input})
                    st.write(response["output"])
                    st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error processing your request {str(e)}")
    else:
        st.warning("Please enter a question first.")

