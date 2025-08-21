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

def get_stock_data_2y(ticker: str):
    data = yf.download(ticker, period="2y", interval="1mo")
    return data.to_string()

def get_stock_data_5y(ticker: str):
    data = yf.download(ticker, period="5y", interval="1mo")
    return data.to_string()

def get_stock_data_10y(ticker: str):
    data = yf.download(ticker, period="10y", interval="1mo")
    return data.to_string()


def get_stock_metrics(ticker_symbol: str):
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info
    metrics = {
        "Market Cap": info.get("marketCap"),
        "P/E Ratio": info.get("trailingPE"),
        "Forward P/E": info.get("forwardPE"),
        "Price/Book": info.get("priceToBook"),
        "Dividend Yield": info.get("dividendYield"),
        "Beta": info.get("beta"),
        "Earnings Growth (Quarterly)": info.get("earningsQuarterlyGrowth"),
        "Recommendation": info.get("recommendationKey")  # buy/hold/sell
    }
    return str(metrics)

yfinance_2y_tool = Tool(
    name="YFinance Short-Term Stock Data",
    func=get_stock_data_2y,
    description="Fetch 2-year monthly stock data for short-term investment analysis."
)

yfinance_5y_tool = Tool(
    name="YFinance Mid-Term Stock Data",
    func=get_stock_data_5y,
    description="Fetch 5-year monthly stock data for medium-term investment analysis."
)

yfinance_10y_tool = Tool(
    name="YFinance Long-Term Stock Data",
    func=get_stock_data_10y,
    description="Fetch 10-year monthly stock data for long-term investment analysis."
)

yfinance_metrics_tool = Tool(
    name="YFinance Fundamental Metrics",
    func=get_stock_metrics,
    description="Fetch key financial metrics and analyst recommendation for a given ticker symbol, useful for stock evaluation."
)

# 3. LLM
llm = ChatVertexAI(model_name="gemini-2.5-flash", temperature=0)

# 4. Tools list
tools = [search_tool, yfinance_2y_tool, yfinance_5y_tool, yfinance_10y_tool, yfinance_metrics_tool]

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