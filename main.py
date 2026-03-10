# -*- coding: utf-8 -*-
"""
News Analysis 2.0 — FastAPI Backend Server
===========================================

.env Variables Required:
-------------------------
# No mandatory .env variables for core functionality.
# Optional overrides:
#
#   HOST=0.0.0.0          # Server host (default: 0.0.0.0)
#   PORT=8000             # Server port (default: 8000)
#   WORKERS=1             # Uvicorn workers (default: 1)
#
# Note: FinBERT model is loaded from HuggingFace on startup.
# For offline/airgapped environments, pre-download the model and set:
#   FINBERT_MODEL_PATH=./local_finbert   # Path to local FinBERT model
#
# NSE API is called at runtime — no API key needed (public endpoint).
# If behind a proxy, set:
#   HTTP_PROXY=http://your.proxy:port
#   HTTPS_PROXY=http://your.proxy:port
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional


import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

from functions import (
    fetch_market_news,
    weighted_sentiment,
    load_features,
    model_prediction,
    deep_research,
    scan_market,
    get_index,
)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

load_dotenv()

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
FINBERT_MODEL_PATH = os.getenv("FINBERT_MODEL_PATH", "ProsusAI/finbert")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Global model state (loaded once at startup)
# ──────────────────────────────────────────────

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy models once at server startup, release on shutdown."""
    logger.info("Loading FinBERT model from '%s' …", FINBERT_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_PATH)
    nlp_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_PATH)
    nlp_model.eval()
    _state["tokenizer"] = tokenizer
    _state["model_nlp"] = nlp_model
    logger.info("FinBERT loaded ✓  (device: %s)", "cuda" if torch.cuda.is_available() else "cpu")

    # Monkey-patch functions.py globals so every helper uses the shared model
    import functions as fn
    fn.tokenizer = tokenizer
    fn.model_nlp = nlp_model

    yield  # server runs here

    logger.info("Shutting down — releasing model resources.")
    _state.clear()


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────


app = FastAPI(
    title="News Analysis 2.0 API",
    description="Indian stock market sentiment + ML swing-trade signals powered by FinBERT & RandomForest.",
    version="2.0.0",
    lifespan=lifespan,
)


# Allow CORS from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────

class TickerRequest(BaseModel):
    ticker: str  # e.g. "RELIANCE.NS"


class ScanRequest(BaseModel):
    index_name: str = "NIFTY 50"   # e.g. "NIFTY 100", "NIFTY 500"
    limit: Optional[int] = None    # cap number of stocks processed
    top_n: int = 10                # how many top/bottom stocks to return


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/health")
def health():
    """
    Health-check endpoint.

    HOW TO CALL:
        GET /health

    Example (curl):
        curl http://localhost:8000/health

    Example (Python requests):
        import requests
        r = requests.get("http://localhost:8000/health")
        print(r.json())

    Returns:
        {"status": "ok", "model_loaded": true}
    """
    return {"status": "ok", "model_loaded": "model_nlp" in _state}


# ──────────────────────────────────────────────

@app.get("/news/{ticker}")
def get_news(ticker: str):
    """
    Fetch raw company-specific and macro RSS news for a ticker.

    HOW TO CALL:
        GET /news/{ticker}

    Path param:
        ticker  — NSE symbol, e.g. RELIANCE.NS  (include .NS suffix)

    Example (curl):
        curl http://localhost:8000/news/RELIANCE.NS

    Example (Python requests):
        import requests
        r = requests.get("http://localhost:8000/news/RELIANCE.NS")
        print(r.json())

    Returns:
        {
          "ticker": "RELIANCE.NS",
          "company_news": [ {"title": "...", "link": "..."}, ... ],
          "macro_news":   [ {"title": "...", "link": "..."}, ... ]
        }
    """
    try:
        company, macro = fetch_market_news(ticker)
        return {
            "ticker": ticker,
            "company_news": company if company else [],
            "macro_news": macro if macro else [],
        }
    except Exception as exc:
        logger.exception("Error fetching news for %s", ticker)
        raise HTTPException(status_code=500, detail=str(exc))


# ──────────────────────────────────────────────

@app.get("/sentiment/{ticker}")
def get_sentiment(ticker: str):
    """
    Compute weighted FinBERT sentiment for a ticker.
    (company sentiment × 0.7) + (macro sentiment × 0.3)

    HOW TO CALL:
        GET /sentiment/{ticker}

    Path param:
        ticker  — e.g. TCS.NS

    Example (curl):
        curl http://localhost:8000/sentiment/TCS.NS

    Example (Python requests):
        import requests
        r = requests.get("http://localhost:8000/sentiment/TCS.NS")
        print(r.json())

    Returns:
        {
          "ticker": "TCS.NS",
          "company_sentiment": 0.12,
          "macro_sentiment": -0.05,
          "weighted_sentiment": 0.07
        }
    """
    try:
        from functions import sentiment_score
        company, macro = fetch_market_news(ticker)
        comp_score = sentiment_score(company)
        mac_score  = sentiment_score(macro)
        weighted   = weighted_sentiment(company, macro)
        return {
            "ticker": ticker,
            "company_sentiment": round(float(comp_score), 4),
            "macro_sentiment":   round(float(mac_score), 4),
            "weighted_sentiment": round(float(weighted), 4),
        }
    except Exception as exc:
        logger.exception("Error computing sentiment for %s", ticker)
        raise HTTPException(status_code=500, detail=str(exc))


# ──────────────────────────────────────────────

@app.get("/technical/{ticker}")
def get_technical(ticker: str):
    """
    Return the latest technical indicators for a ticker
    (RSI, MACD, MA50, MA200, Volume MA, daily return).

    HOW TO CALL:
        GET /technical/{ticker}

    Path param:
        ticker  — e.g. INFY.NS

    Example (curl):
        curl http://localhost:8000/technical/INFY.NS

    Example (Python requests):
        import requests
        r = requests.get("http://localhost:8000/technical/INFY.NS")
        print(r.json())

    Returns:
        {
          "ticker": "INFY.NS",
          "close": 1452.3,
          "rsi": 58.2,
          "macd": 12.1,
          "macd_signal": 10.5,
          "ma50": 1420.0,
          "ma200": 1380.0,
          "volume_ma": 8200000,
          "daily_return": 0.0034,
          "trend": "UPTREND"
        }
    """
    try:
        df = load_features(ticker)
        if df is None or len(df) < 200:
            raise HTTPException(status_code=422, detail="Not enough historical data for this ticker.")
        latest = df.iloc[-1]
        trend = "UPTREND" if latest["MA50"] > latest["MA200"] else "DOWNTREND"
        return {
            "ticker": ticker,
            "close":        round(float(latest["Close"]), 2),
            "rsi":          round(float(latest["RSI"]), 2),
            "macd":         round(float(latest["MACD"]), 4),
            "macd_signal":  round(float(latest["MACD_signal"]), 4),
            "ma50":         round(float(latest["MA50"]), 2),
            "ma200":        round(float(latest["MA200"]), 2),
            "volume_ma":    round(float(latest["VOL_MA"]), 0),
            "daily_return": round(float(latest["Return"]), 6),
            "trend":        trend,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error fetching technical data for %s", ticker)
        raise HTTPException(status_code=500, detail=str(exc))


# ──────────────────────────────────────────────

@app.get("/predict/{ticker}")
def predict(ticker: str):
    """
    Run the RandomForest ML model and return a bullish probability
    combined with live FinBERT sentiment.

    HOW TO CALL:
        GET /predict/{ticker}

    Path param:
        ticker  — e.g. HDFCBANK.NS

    Example (curl):
        curl http://localhost:8000/predict/HDFCBANK.NS

    Example (Python requests):
        import requests
        r = requests.get("http://localhost:8000/predict/HDFCBANK.NS")
        print(r.json())

    Returns:
        {
          "ticker": "HDFCBANK.NS",
          "bullish_probability": 0.72,
          "signal": "BULLISH SWING",
          "weighted_sentiment": 0.08
        }
    """
    try:
        df = load_features(ticker)
        if df is None or len(df) < 200:
            raise HTTPException(status_code=422, detail="Not enough historical data.")
        company, macro = fetch_market_news(ticker)
        sent = weighted_sentiment(company, macro)
        prob = model_prediction(df, sent)

        signal = "SIDEWAYS"
        if prob > 0.6:
            signal = "BULLISH SWING"
        elif prob < 0.4:
            signal = "BEARISH SWING"

        return {
            "ticker": ticker,
            "bullish_probability": round(float(prob), 4),
            "signal": signal,
            "weighted_sentiment": round(float(sent), 4),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error predicting for %s", ticker)
        raise HTTPException(status_code=500, detail=str(exc))


# ──────────────────────────────────────────────

@app.get("/research/{ticker}")
def research(ticker: str):
    """
    Full deep-research report for a single stock:
    technicals + FinBERT news sentiment + ML signal + swing-trade plan.

    HOW TO CALL:
        GET /research/{ticker}

    Path param:
        ticker  — e.g. SBIN.NS

    Example (curl):
        curl http://localhost:8000/research/SBIN.NS

    Example (Python requests):
        import requests
        r = requests.get("http://localhost:8000/research/SBIN.NS")
        print(r.json())

    Returns:
        {
          "ticker": "SBIN.NS",
          "technical":    { "trend": "UPTREND", "rsi": 61.2, "volatility": "1.4%" },
          "news_analysis": {
              "company_specific": [ {"title": "...", "snippet": "...", "link": "..."} ],
              "market_macro":     [ {"title": "...", "link": "..."} ]
          },
          "ai_probability": { "bullish_probability": "68.0%", "signal": "BULLISH SWING" },
          "swing_plan":    { "entry": 812.5, "stoploss": 788.1, "target": 853.1 }
        }
    """
    try:
        report = deep_research(ticker)
        return {"ticker": ticker, **report}
    except Exception as exc:
        logger.exception("Error in deep research for %s", ticker)
        raise HTTPException(status_code=500, detail=str(exc))


# ──────────────────────────────────────────────    

@app.post("/scan")
def scan(req: ScanRequest):
    """
    Scan an entire NSE index, rank all stocks by bullish probability,
    and return the top N bullish and top N bearish picks.

    HOW TO CALL:
        POST /scan
        Content-Type: application/json

    Request body (JSON):
        {
          "index_name": "NIFTY 50",   // "NIFTY 100" | "NIFTY 500" etc.
          "limit": 20,                // optional — process only first N stocks
          "top_n": 10                 // how many results to return per side
        }

    Example (curl):

        curl -X POST http://localhost:8000/scan \\
             -H "Content-Type: application/json" \\
             -d '{"index_name": "NIFTY 50", "top_n": 5}'

    Example (Python requests):
        import requests
        payload = {"index_name": "NIFTY 50", "limit": 20, "top_n": 5}
        r = requests.post("http://localhost:8000/scan", json=payload)
        print(r.json())

    Returns:
        {
          "scan_details": { "index": "NIFTY 50", "stocks_scanned": 48 },
          "top_most_bullish_stocks": [ {"stock": "...", "bullish_probability": "78%"} ],
          "top_most_bearish_stocks": [ {"stock": "...", "bearish_strength": "65%"} ]
        }
    """
    try:
        result = scan_market(
            index_name=req.index_name,
            limit=req.limit,
            top_n=req.top_n,
        )
        return result
    except Exception as exc:
        logger.exception("Error during market scan")
        raise HTTPException(status_code=500, detail=str(exc))


# ──────────────────────────────────────────────

@app.get("/index")
def index_stocks(
    index_name: str = Query(default="NIFTY 50", description="NSE index name, e.g. 'NIFTY 100'")
):
    """
    Return the list of ticker symbols in a given NSE index.

    HOW TO CALL:
        GET /index?index_name=NIFTY%2050

    Query params:
        index_name  — default "NIFTY 50" (URL-encode the space as %20 or +)

    Example (curl):
        curl "http://localhost:8000/index?index_name=NIFTY+50"

    Example (Python requests):
        import requests
        r = requests.get("http://localhost:8000/index", params={"index_name": "NIFTY 100"})
        print(r.json())

    Returns:
        {
          "index": "NIFTY 50",
          "count": 50,
          "symbols": ["RELIANCE.NS", "TCS.NS", ...]
        }
    """
    try:
        symbols = get_index(index_name)
        return {"index": index_name, "count": len(symbols), "symbols": symbols}
    except Exception as exc:
        logger.exception("Error fetching index %s", index_name)
        raise HTTPException(status_code=500, detail=str(exc))


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )