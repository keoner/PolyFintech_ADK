from langchain.agents import initialize_agent
from langchain_google_vertexai import ChatVertexAI
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper

import yfinance as yf
import pandas as pd

from dotenv import load_dotenv
import os

import datetime as dt

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

def get_technical_indicators(ticker_symbol: str):
    data = yf.download(ticker_symbol, period="1y", interval="1d")
    
    # Moving Averages
    data['MA50'] = data['Close'].rolling(50).mean()
    data['MA200'] = data['Close'].rolling(200).mean()
    
    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Keep only the last 5 rows to summarize recent trend
    summary = data[['Close', 'MA50', 'MA200', 'RSI', 'MACD', 'Signal']].tail(5)
    
    return summary.to_string()

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
        "Recommendation": info.get("recommendationKey")
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

technical_tool = Tool(
    name="Technical Indicators",
    func=get_technical_indicators,
    description="Fetch recent stock prices with technical indicators (MA50, MA200, RSI, MACD)"
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
    temperature=0,
    max_iterations=5,
    handle_parsing_errors=True,
)

Industry = "Artificial Intelligence"
investment_horizon = "short"
date = dt.datetime.now().strftime("%Y-%m-%d")

from langchain.callbacks import StdOutCallbackHandler
callback = StdOutCallbackHandler()

inputs = {"input": f"""
You are a financial research assistant. Today's date is {date}.
Your task is to recommend a single stock for investment in the {Industry} industry based on the user's investment horizon ({investment_horizon} term). Use the following steps:

1. Identify 2-3 top publicly traded companies in the industry using Google and other sources.
2. Gather key information for each company:
   - Recent news and announcements
   - Analyst forecasts and ratings
   - Market sentiment
   - Company strategies and initiatives
3. Collect historical stock data from yfinance:
 - Use monthly data for the past year to assess recent trends.
 - Evaluate volatility, recent performance, and growth potential.
 - Compute and analyze technical indicators, including:
   - Moving averages (MA50, MA200)
   - Relative Strength Index (RSI)
   - MACD
   - Average True Range (ATR)
4. Compare at least 2-3 stocks from the industry using all gathered data, including fundamentals, news, and technical indicators.
5. Select the single strongest candidate and provide a clear rationale for your choice.

Format your findings like this:
- Company Name:
- Why this company is a good pick:
- Company strategies and initiatives:
- Management Team quality:
- Stock Performance (last 12 months monthly data and technical indicators):
- Technical Indicators Analysis (MA50 vs MA200, RSI trend, MACD signals, ATR overview):
- Final Recommendation:

**Provide only the final recommendation and reasoning. No extra steps.**
"""}

result = agent.invoke(input=inputs, callbacks=[callback], return_intermediate_steps=True)

final_output = result.get("output") or result.get("result") or "No output found."

from rich.console import Console
from rich.markdown import Markdown

console = Console()
md = Markdown(final_output)
console.print(md)