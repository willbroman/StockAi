import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
import plotly.graph_objects as go
import datetime
import requests
from bs4 import BeautifulSoup

def fetch_stock_data(ticker):
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=60)
    data = yf.download(ticker, start=start_date, end=end_date, group_by='ticker')
    data.reset_index(inplace=True)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in data.columns]
    return data

def fetch_recent_news(ticker):
    query = f"{ticker} stock"
    url = f"https://www.google.com/search?q={query}&tbm=nws"
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')
    headlines = []
    for g in soup.find_all('div', class_='BNeawe vvjwJb AP7Wnd'):
        headlines.append(g.get_text())
        if len(headlines) >= 5:
            break
    return headlines

def prepare_prophet_data(df, ticker):
    close_col = f'{ticker}_Close'
    if 'Date' not in df or close_col not in df:
        raise ValueError(f"Expected columns 'Date' and '{close_col}' not found in data")
    df = df[['Date', close_col]].dropna()
    df = df.rename(columns={'Date': 'ds', close_col: 'y'})
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['y'])
    return df

def make_forecast(df):
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast

def plot_forecast(df, forecast, ticker):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    fig.add_trace(go.Scatter(
        x=forecast['ds'].tolist() + forecast['ds'][::-1].tolist(),
        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(192,192,192,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval',
        showlegend=True
    ))

    fig.update_layout(
        title=f"Forecast for {ticker}",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified"
    )

    return fig

def make_recommendation(forecast):
    last_actual = forecast.loc[forecast['ds'] == forecast['ds'].max() - pd.Timedelta(days=30), 'yhat'].values
    next_month_mean = forecast.loc[forecast['ds'] > forecast['ds'].max() - pd.Timedelta(days=1), 'yhat'].mean()
    if len(last_actual) == 0:
        return "Hold (insufficient data)"
    last = last_actual[0]
    change = (next_month_mean - last) / last
    if change > 0.05:
        return "Buy (expected rise >5%)"
    elif change < -0.05:
        return "Sell (expected drop >5%)"
    else:
        return "Hold (expected stable price)"

# ---------------------
# Streamlit Web UI
# ---------------------
st.title("📈 Stock Forecast AI")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL, TSLA, MSFT):").upper()

if ticker:
    st.subheader(f"Fetching data for {ticker}...")

    try:
        stock_data = fetch_stock_data(ticker)
        prophet_df = prepare_prophet_data(stock_data, ticker)

        st.subheader("📰 Recent News")
        news = fetch_recent_news(ticker)
        for idx, headline in enumerate(news, 1):
            st.write(f"{idx}. {headline}")

        st.subheader("📊 Forecast")
        forecast = make_forecast(prophet_df)
        fig = plot_forecast(prophet_df, forecast, ticker)
        st.plotly_chart(fig)

        st.subheader("💡 Recommendation")
        recommendation = make_recommendation(forecast)
        st.write(recommendation)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
