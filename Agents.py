from dotenv import load_dotenv
import os
import requests
import streamlit as st

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from tavily import TavilyClient
from langchain.agents import create_agent


# =========================
# Load Environment Variables
# =========================

load_dotenv()


# =========================
# 🌦️ Weather Tool
# =========================

@tool
def get_weather(city: str) -> str:
    """Get current weather of a city."""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q={city},IN&appid={api_key}&units=metric"
    )

    response = requests.get(url)
    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {desc}, {temp}°C"


# =========================
# 📰 News Tool
# =========================

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


@tool
def get_news(city: str) -> str:
    """Get latest news about a city."""

    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    results = response.get("results", [])

    if not results:
        return f"No news found for {city}"

    news_list = []

    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")

        news_list.append(
            f"- {title}\n"
            f"  🔗 {url}\n"
            f"  📝 {snippet[:200]}..."
        )

    return (
        f"Latest news in {city}:\n\n"
        + "\n\n".join(news_list)
    )


# =========================
# 🧠 LLM
# =========================

llm = ChatMistralAI(
    model="mistral-small-2506"
)


# =========================
# 🤖 Agent
# =========================

agent = create_agent(
    llm,
    tools=[get_weather, get_news],
    system_prompt="You are a helpful city assistant."
)


# =========================
# 🎨 Streamlit UI
# =========================

st.set_page_config(
    page_title="City AI Assistant",
    page_icon="🌆",
    layout="centered"
)

st.title("🌆 City AI Assistant")
st.write(
    "Ask me about weather or latest news of any city in India."
)


# =========================
# Chat History
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =========================
# User Input
# =========================

user_input = st.chat_input(
    "Ask something like: What's the weather in Pune?"
)


if user_input:

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)


    # Agent response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            result = agent.invoke({
                "messages": [
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            })

            response = result["messages"][-1].content

            st.markdown(response)


    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
