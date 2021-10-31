#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Description


# In[17]:


#App builder
import streamlit as st 
from PIL import Image 

#Visualization
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

#Mathematics
import pandas as pd 
import numpy as np

#Timeseries dataframe
from TimeSeriesDataSet import df


# In[18]:


DATA_URL = df
st.markdown("# Stochastic Gradient Descent in Continuous time- By Joseph Bunster")
st.markdown("Explore the dataset to know more about OT based Financial Time series Dataset Linear Regression")
img=Image.open('images/carbon.png')
st.image(img,width=700)
st.markdown('''
**Abstract**
Stochastic gradient descent in continuous time **(SGDCT)** provides a computationally eficient method for the statistical learning of continuous-Time models, which are widely used in science, engineering, and **finance**. 
The SGDCT algorithm follows a (noisy) descent direction along a continuous stream of data. SGDCT performs an online parameter update in continuous time with the parameter updates **θt**, satisfying a stochastic diffierential equation. 
We prove that **limδ rg(θ) = 0**, where **g** is anatural objective function for the estimation of the continuous-Time dynamics. 
The convergence proof leverages ergodicity by using an appropriate Poisson equation to help describe the evolution of the parameters for large times. 
For certain continuous-Time problems, SGDCT has some promising advantages compared to a traditional stochastic gradient descent algorithm. 
This paper mainly focuses **on applications in finance**, such as model estimation for stocks, bonds, interest rates, and financial derivatives. 
SGDCT can also be used for the optimization of high-dimensional continuous-time models, such as American options. As an example application, SGDCT is combined with a deep neural network to price high-dimensional American options (up to 100 dimensions).
Author keywords
*American options; Deep learning; Machine learning; Statistical learning; Stochastic difierential equations; Stochastic gradient descent*
source: https://www.scopus.com/record/display.uri?eid=2-s2.0-85041577966&origin=inward&txGid=b958c8de8483660591ca27e29596028b
''')
st.markdown("The data presented are of 5 different companies - **Microsoft, Apple, Tesla, Google and Amazon,** collected from Tiingo API **https://www.tiingo.com.**")


# In[19]:


if st.button("Learn more about Joseph Bunster and data processed"):
    img=Image.open('images/author.png')
    st.markdown("**Joseph Bunster ** Joe Bunster is hardworking mathematics professional, passionate about applying my technical background to solving real world problems.I enjoy challenges and thrive under pressure, these traits helped me successfully compete in the 2018 US National Collegiate Mathematics Championship where I placed 3rd in the United States.Currently enrolled in a Masters of Science in Mathematics Program at NYU-Courant, with an expected graduation date of May 2021. I have previous research experience in Financial Engineering,Optimal Control Theory, and Reinforcement Learning.Interests in Applied Math, Probability, Optimization and Finance..")
    st.image(img,width=200, caption="Joe Bunster 🤵‍")
    st.markdown("The data was collected and made available by **[Joseph Bunster](https://www.linkedin.com/in/joseph-bunster/)**.")
    images=Image.open('images/tiingo.png')
    st.image(images,width=600)
    #Ballons
    st.balloons()


# In[20]:


st.info(''' The optimal transport theory is the study of optimal transportation and allocation between measures.
The optimal transport problem was first introduced by Monge (1781) and formalized by Kantorovitch (1942), leading to the so called Monge-Kantorovitch transportation problem.
The goal is to look for a transport map transforming a probability density function into another while minimizing the cost of transport.
''')
img=Image.open("images/OT.jpg")
st.image(img,width=700)


# In[25]:


st.sidebar.markdown("## Side Panel")
st.sidebar.markdown("Use this panel to explore the dataset and create own viz.")

st.header("Now, Explore Yourself the Time Series Dataset")

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading timeseries dataset...')

# Load 10,000 rows of data into the dataframe.
df 
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading timeseries dataset...Completed!')
images=Image.open('images/bot.png')
st.image(images,width=150)


# In[26]:


# Showing the original raw data
if st.checkbox("Show Raw Data", False):
    st.subheader('Raw data')
    st.write(df)
    
    
st.title('Quick  Explore')
st.sidebar.subheader(' Quick  Explore')
st.markdown("Tick the box on the side panel to explore the dataset.")


if st.sidebar.checkbox('Basic info'):
    if st.sidebar.checkbox('Quick Look'):
        st.subheader('Dataset Quick Look:')
        st.write(df.head())
    if st.sidebar.checkbox("Show Columns"):
        st.subheader('Show Columns List')
        all_columns = df.columns.to_list()
        st.write(all_columns)
   
    if st.sidebar.checkbox('Statistical Description'):
        st.subheader('Statistical Data Descripition')
        st.write(df.describe())
    if st.sidebar.checkbox('Missing Values?'):
        st.subheader('Missing values')
        st.write(df.isnull().sum())
if st.sidebar.checkbox('Dataset Another Quick Look'):
    st.subheader('Dataset Quick Look:')
    st.write(df.head())


# In[27]:


st.title('Create Own Visualization')
st.markdown("Tick the box on the side panel to create your own Visualization.")
st.sidebar.subheader('Create Own Visualization')
if st.sidebar.checkbox('Graphics'):
    if st.sidebar.checkbox('Count Plot'):
        st.subheader('Count Plot')
        st.info("If error, please adjust column name on side panel.")
        column_count_plot = st.sidebar.selectbox("Choose a column to plot count. Try Selecting Sex ",df.columns)
        hue_opt = st.sidebar.selectbox("Optional categorical variables (countplot hue). Try Selecting Species ",df.columns.insert(0,None))
        
        fig = sns.countplot(x=column_count_plot,data=df,hue=hue_opt)
        st.pyplot()
            
            
    if st.sidebar.checkbox('Histogram | Distplot'):
        st.subheader('Histogram | Distplot')
        st.info("If error, please adjust column name on side panel.")
        # if st.checkbox('Dist plot'):
        column_dist_plot = st.sidebar.selectbox("Optional categorical variables (countplot hue). Try Selecting Body Mass",df.columns)
        fig = sns.distplot(df[column_dist_plot])
        st.pyplot()
            
            
 
        
    if st.sidebar.checkbox('Boxplot'):
        st.subheader('Boxplot')
        st.info("If error, please adjust column name on side panel.")
        column_box_plot_X = st.sidebar.selectbox("X (Choose a column). Try Selecting island:",df.columns.insert(0,None))
        column_box_plot_Y = st.sidebar.selectbox("Y (Choose a column - only numerical). Try Selecting Body Mass",df.columns)
        hue_box_opt = st.sidebar.selectbox("Optional categorical variables (boxplot hue)",df.columns.insert(0,None))
        # if st.checkbox('Plot Boxplot'):
        fig = sns.boxplot(x=column_box_plot_X, y=column_box_plot_Y,data=df,palette="Set3")
        st.pyplot()


# In[28]:


st.sidebar.markdown("[Data Source](https://www.ttingo.com)")
st.sidebar.info("Linkedin [Joseph Bunster](https://www.linkedin.com/in/joseph-bunster/) ")
st.sidebar.info("Self Exploratory Visualization using Optimal Transport on Financial Time Series Data- Brought To you By [Jospeh Bunster](https://github.com/Joseph-Bunster)  ")
st.sidebar.text("Built with  ❤️ by Joe Bunster")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




