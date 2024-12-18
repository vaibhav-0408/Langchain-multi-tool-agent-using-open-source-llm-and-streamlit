import os
import datetime
import streamlit as st
import wikipediaapi
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ChatMessageHistory


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


st.title("Multi-Tool Agent with LLaMA-3 via Groq")


llm = ChatGroq(groq_api_key=groq_api_key, model='llama3-8b-8192')


wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent="agent_tools/1.0 (vaibhav.dimble@thinkitive.com)" 
)

def search_wikipedia(query):
    try:
        print(f"[Tool Called: Wikipedia] Searching for: {query}")
        page = wiki_wiki.page(query)
        if not page.exists():
            return f"Sorry, no article found for {query}."
        return page.summary[:500]  
    except Exception as e:
        return f"Error: {e}"


search_tool = Tool(
    name="Wikipedia Search",
    func=search_wikipedia,
    description="Use this tool to search Wikipedia for articles on any topic."
)
def calculate(expression):
    try:
        print(f"[Tool Called: Calculation] Calculating expression: {expression}")
        result = eval(expression)
        return f"The result of the calculation is: {result}"
    except Exception as e:
        return f"Error in calculation: {e}"

calculation_tool = Tool(
    name="Calculator",
    func=calculate,
    description="Use this tool to perform arithmetic calculations. Example: '2 + 2 * 3'"
)


def get_datetime(query=None):
    print(f"[Tool Called: DateTime] Getting current date and time.")
    return f"The current date and time is: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

datetime_tool = Tool(
    name="DateTime",
    func=get_datetime,
    description="Use this tool to get the current date and time."
)


tools = [search_tool, calculation_tool, datetime_tool]


agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  
    verbose=True
)



if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()



user_query = st.text_input("Ask me anything (Search, Calculate, or Get DateTime):")

if st.button("Get Response"):
    if user_query:
        with st.spinner("Thinking..."):
            try:
          
                st.session_state.chat_history.add_user_message(user_query)

              
                response = agent.run(user_query)
                
                
                st.session_state.chat_history.add_ai_message(response)
                
            
                st.success(response)
                
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question to proceed.")


st.write("### Conversation History")
for msg in st.session_state.chat_history.messages:
    st.write(f"**{msg.type.capitalize()}**: {msg.content}")


if st.button("Clear Conversation"):
    st.session_state.chat_history = ChatMessageHistory()
    st.success("Chat history cleared.")
