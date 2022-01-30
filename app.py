import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
from cryptocmd import CmcScraper

START = '2015-01-01'
TODAY = date.today().strftime('%Y-%m-%d')

st.title('株価、仮想通貨価格予測アプリ(develop)')

selected_stocks = st.sidebar.text_input("証券コード、ティッカーシンボル(AAPL、GOOG等)を入力して下さい。※初期値は日経225", "^N225")

if selected_stocks.isdecimal():
    selected_stocks = selected_stocks + ".T"

n_years = st.slider('何年後を予測しますか？:', 1, 4)
period = n_years *365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('データを読み込んでいます...')
data = load_data(selected_stocks)
data_load_state.text('データを読み込みました!')

st.subheader('株価')
st.write(data.set_index('Date').tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text='Time Series Data（株価）', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forcasting
### Use button to start model prediction
if st.button("予測出力（株価）"):
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={'Date': 'ds', 'Close': 'y'})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    # forecast = forecast.set_index('Date')

    st.subheader('予測株価')
    st.write(forecast.tail())

    st.write('forcast data')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write('forcast components')
    fig2 = m.plot_components(forecast)
    st.write(fig2)

### Select ticker
st.subheader('仮想通貨価格（ドル換算）')
selected_ticker = st.sidebar.text_input("仮想通貨の通貨名を入力して下さい。(BTC, ETH, LINK等)", "BTC")

### Initialise scraper
@st.cache
def load_data(selected_ticker):
	init_scraper = CmcScraper(selected_ticker)
	df = init_scraper.get_dataframe()
	return df

### Load the data
data_load_state = st.text('データを読み込んでいます...')
data = load_data(selected_ticker)
data_load_state.text('データを読み込みました!')

### Get the scraped data
scraper = CmcScraper(selected_ticker)
data = scraper.get_dataframe()

### Create overview of the latest rows of data
st.write(data.sort_values('Date').set_index('Date').tail())

### Plot functions for regular & log plots
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
	fig.layout.update(title_text='Time Series data（仮想通貨価格）', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

def plot_raw_data_log():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
	fig.update_yaxes(type="log")
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
### Create checkbox for plotting (log) data
plot_log = st.checkbox("Plot log scale")
if plot_log:
	plot_raw_data_log()
else:
	plot_raw_data()

### Use button to start model prediction
if st.button("予測出力（仮想通貨価格）"):

	### Get the required data & rename the columns so fbprophet can read it
	df_train = data[['Date','Close']]
	df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

	### Create Prophet model
	m = Prophet(
		changepoint_range=0.8, # percentage of dataset to train on
		yearly_seasonality='auto', # taking yearly seasonality into account
		weekly_seasonality='auto', # taking weekly seasonality into account
		daily_seasonality=False, # taking daily seasonality into account
		seasonality_mode='multiplicative' # additive (for more linear data) or multiplicative seasonality (for more non-linear data)
	)
	
	m.fit(df_train)
    
	### Predict using the model
	future = m.make_future_dataframe(periods=365)
	forecast = m.predict(future)

	### Show and plot forecast
	st.subheader('予測価格（仮想通貨）')
	st.write(forecast.set_index('ds').tail())

	st.subheader(f'Forecast plot for 365 days')
	fig1 = plot_plotly(m, forecast)
	if plot_log:
		fig1.update_yaxes(type="log")
	st.plotly_chart(fig1)

	st.subheader("Forecast components")
	fig2 = m.plot_components(forecast)
	st.write(fig2)