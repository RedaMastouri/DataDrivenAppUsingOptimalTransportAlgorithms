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
print("Today's date:", today)
print("Exactly 6 months date:", six_months_ago)


# ### Imlepenting tiingo.com:
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


for ticker in TICKERS:
    #Dynamically create Data frames
    vars()[ticker] = pdr.get_data_tiingo(ticker, start, end, api_key=token)


# In[6]:


TICKERS_Frames = [ AAPL, MSFT, Goog, AMZN, TSLA]
def returned_dataFrame(list_of_frame):
    df = pd.concat(list_of_frame)
    return df


# In[8]:


df = returned_dataFrame(TICKERS_Frames)
df.head()


# 
# 
