from langchain.agents import initialize_agent
from langchain_google_vertexai import ChatVertexAI
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper

import yfinance as yf
import pandas as pd

from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatVertexAI(model_name="gemini-2.5-flash", temperature=0)
search = GoogleSearchAPIWrapper()

search_tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run,
)

def get_stock_data(ticker: str):
    data = yf.download(ticker, period="1y", interval="1mo")
    return data.to_string()

yfinance_tool = Tool(
    name="YFinance Stock Data",
    func=get_stock_data,
    description="Fetch 1-year stock data for a given ticker symbol."
)

# 3. LLM
llm = ChatVertexAI(model_name="gemini-2.5-flash", temperature=0)

# 4. Tools list
tools = [search_tool, yfinance_tool]

# 5. Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True,
)

Industry = "Artificial Intelligence"
investment_horizon = "short"

result = agent.run(f"""
You are a financial research assistant. I will give you one industry and an investment horizon.
Your task is to help a user choose a single stock to invest in, based on the following steps:

1. Search for the top publicly traded companies in the given industry using Google.
2. Gather relevant news, analyst forecasts, market sentiment, and company strategies that might affect their stock performance.
3. Compare at least 2-3 stocks using both the Google information and historical stock data from yfinance.
   - Use monthly data for the past year to understand trends.
   - Consider volatility, recent performance, and growth potential.
4. Evaluate the stocks based on the user's investment horizon ({investment_horizon} term).
   - Short-term: focus on recent momentum and near-term catalysts.
   - Medium-term: balance growth and stability.
   - Long-term: focus on fundamentals, market position, and potential for sustained growth.
5. Pick one stock that is the strongest candidate for this investment horizon and explain clearly why you chose it.
6. Format your findings using this structure:
   - Company Name:
   - Why this company is a good pick:
   - What the company is doing / strategies:
   - Management Team quality:
   - Stock Performance (last 12 months monthly data):
   - Final Recommendation:
The industry is: {Industry}.
""")
print(result)