#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Here add a description


# In[2]:


'''
Importing all the needed librairies
'''
#Data Structure, scientific computing and technical computing.
import numpy as np
import pandas as pd
import pandas_datareader.data as web # pip install pandas_datareader
#Dataframe
import pandas_datareader as pdr

#Scipy: scientific computing
import scipy
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance_matrix
from scipy import stats


#Visualization
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning library for the Python programming language. 
from sklearn.neighbors import KernelDensity

#Dataset
import yfinance as yf #pip install yfinance

#Date formatting
#Today's date
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta


#Statistics
import scipy



#Past six month function
def past_six_month_date():
    #librairies
    from datetime import date, timedelta
    from dateutil.relativedelta import relativedelta
    #emptylist
    mylist = []
    #Processing six months back
    sixp = date.today() - relativedelta(months=+6)
    mylist.append(sixp)
    return f"{mylist[0]:%Y-%m-%d}"


six_months_ago = past_six_month_date()
today = date.today()

#For printing results
#print("Today's date:", today)
#print("Exactly 6 months date:", six_months_ago)


# ### Implementing tiingo.com:
# A financial research platform dedicated to creating innovative financial tools for all, while adopting the motto, **"Actively Do Good"**.
# 

# In[3]:


Login = 'bunster'
pw = 'M@$t0ur1'
start = six_months_ago
end = today
TICKERS = [ 'AAPL','MSFT', 'Goog', 'AMZN', 'TSLA']
apiURL= 'https://api.tiingo.com/documentation/end-of-day'
token = '2d10bb042e786244063efc000e6dc15e79b07274'


# In[4]:


def get_adjusted_close(ticker, start, end, token):
    import pandas_datareader as pdr
    df = pdr.get_data_tiingo(ticker, start, end, api_key=token)
    return df


# In[5]:


def returned_dataFrame(list_of_frame):
    df = pd.concat(list_of_frame)
    return df


# In[6]:


def build_dataset(ticker):
    from tiingo import TiingoClient
    config = {}
    # To reuse the same HTTP Session across API calls (and have better performance),
    config['session'] = True
    # If you don't have your API key as an environment variable,
    # pass it in via a configuration dictionary.
    config['api_key'] = "2d10bb042e786244063efc000e6dc15e79b07274"
    # Initialize
    client = TiingoClient(config)
    
    
    
    df = client.get_dataframe(ticker, startDate = six_months_ago, endDate= today, frequency='daily', metric_name=None)
    return df


# In[7]:



#df = build_dataset('AAPL')


# In[8]:


TICKERS_Frames = ['AAPL', 'MSFT', 'Goog', 'AMZN', 'TSLA']
AAPL = build_dataset(TICKERS_Frames[0]) 
MSFT = build_dataset(TICKERS_Frames[1]) 
Goog = build_dataset(TICKERS_Frames[2]) 
AMZN = build_dataset(TICKERS_Frames[3]) 
TSLA = build_dataset(TICKERS_Frames[4]) 


AAPL['Ticker']= 'AAPL'
MSFT['Ticker']= 'MSFT'
Goog['Ticker']= 'Goog'
AMZN['Ticker']= 'AMZN'
TSLA['Ticker']= 'TSLA'

Frames = [ AAPL, MSFT, Goog, AMZN, TSLA]
df = pd.concat(Frames)


# In[ ]:





# 
# 
