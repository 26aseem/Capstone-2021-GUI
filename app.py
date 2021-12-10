# Import the required packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import streamlit as st
import pickle
from pickle import load
from PIL import Image
import seaborn as sns
import statsmodels.api as sm


### Streamlit code starts here    
st.title("Time Series Analysis of Disaster Tweets")
st.markdown("The dashboard will help the government and humanitarian aid agencies to plan and coordinate the natural disaster relief efforts, resulting in more people being saved and more effective distribution of emergency supplies during a natural hazard")
st.sidebar.title("Select Visual Charts")
st.sidebar.markdown("Select the Charts/Plots accordingly:")

# Some CSS Markdown for styling
STYLE = """
<style>
img {
     max-width: 100%;     
}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


### Time Series Code goes here

# Dataset
# Load the Dataset
tweets1 = pd.read_csv("https://raw.githubusercontent.com/anidevhere/Temp/main/nepal_mix_1.csv")[['text','type']]
tweets2 = pd.read_csv("https://raw.githubusercontent.com/anidevhere/Temp/main/italy_mix_1.csv")[['text','type']]
tweets3 = pd.read_csv("https://raw.githubusercontent.com/anidevhere/Temp/main/Covid-19.csv")[['text','type']]
names = [tweets1,tweets2,tweets3]

# Concatenate the datasets
tweets = pd.concat(names,ignore_index = True)

# Reshuffle the dataset
tweets = tweets.sample(frac = 1)

# Reindex the dataset
tweets['index'] = list(range(0,tweets.shape[0],1))
tweets.set_index('index', inplace=True)
tweets['type'] = tweets['type'].map({0: 'Need', 1: 'Availability', 2: 'Other'})

# Change column names for consistency
tweets.columns = ['text', 'type']

print('Shape of the Dataset:',tweets.shape)

# Dataset Description
h = st.sidebar.slider('Select the number of tweets using the slider', 1, tweets.shape[0], 10)
data_tweets = tweets.sample(h)
data_tweets['index'] = list(range(0, h, 1))
data_tweets.set_index('index', inplace=True)
st.table(data_tweets)

# Checking for class balancing and get unique labels:
chart_visual_class_balancing = st.sidebar.checkbox('Class Labels', True)
if chart_visual_class_balancing==True:
    fig = plt.figure(figsize=(8, 4))
    sns.countplot(y=tweets.loc[:, 'type'],data=tweets).set_title("Count of tweets in each class")
    st.pyplot(fig)
  
tweets['type'] = tweets['type'].map({'Need':0, 'Availability':1,'Other':2})

# Get all the labels used in the labelling column
label = tweets.type.unique()
print("Labels:", label)

# Remove label 2 from the list because not required for time series analysis
label = np.delete(label,np.where(label == 2))
print("Labels:", label)

# Add names to the numerical labels
label_name = []
for i in label:
    if i == 0:
        label_name.append("Need")
    elif i == 1:
        label_name.append("Availability")
        
# Choose interval
interval = 30
start_date = "2021-04-01"

# Create Timestamps with intervals
ds = pd.date_range(start=start_date, periods=interval)
dates = []
for i in ds:
    dates.append(i.strftime('%m-%d-%Y'))
del ds

# Divide the Dataset into intervals

# Divide the dataset into the given number of intervals
num_of_tweets_per_interval = math.floor(tweets.shape[0]/interval)

# Create Time Series with intervals
data = []
count_of_data = []
for i in label:
    count_of_data.append([])

for i in range(1,interval+1,1):
    # Draw a sample from the tweets
    tw = tweets.sample(n=num_of_tweets_per_interval, random_state=10, replace=False)
    # Append the statistics of the drawn sample to the list
    stat = dict()
    for j in range(0,len(label)):
        stat[label[j]] = list(tw['type']).count(label[j])
        count_of_data[j].append(list(tw['type']).count(label[j]))
    data.append(stat)
    # Remove the already drawn tweets from the dataset
    tweets.drop(labels=list(tw.index.values),inplace=True)


# Real Time Series starts here
# Load Dataset
df = pd.DataFrame(count_of_data).T
# Set Index
df['Date'] = pd.to_datetime(dates)
df.set_index('Date', inplace=True)
df.columns = ['Need', 'Availability']


st.title("Twitter Data Description")
chart_visual_tweets = st.sidebar.selectbox('Select Chart/Plot type', 
                                    ('Stacked Bar Chart', 'Side-by-Side Bar Chart', 'Line Chart'))

# Plot 1
if chart_visual_tweets=='Side-by-Side Bar Chart':
    # set width of bars
    barWidth = 0.25
    # Set position of bar on X axis
    r = [np.arange(interval)]
    for i in range(1, len(label)):
        r1 = [x + barWidth for x in r[-1]]
        r.append(r1)
    # Plotting a line plot after changing it's width and height
    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(8)
    # Make the plot
    for i,lab in enumerate(label):
        plt.bar(r[i], count_of_data[i], width=barWidth, edgecolor='white', label=label_name[i])
    # Add xticks on the middle of the group bars
    plt.xlabel('Time Series', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(count_of_data[0]))], list(dates))
    plt.tick_params(axis ='x', rotation =90)
    # Create legend & Show graphic
    plt.legend()
    plt.show()    
    st.pyplot(f)

# Plot 2
if chart_visual_tweets=='Stacked Bar Chart':
    # Plotting a line plot after changing it's width and height
    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(8)

    b = np.zeros(interval)
    for i,lab in enumerate(label):
        plt.bar(dates, count_of_data[i],bottom=b, edgecolor='white', label=label_name[i])
        b += np.array(count_of_data[i])
    
    plt.xlabel('Time Series', fontweight='bold')
    plt.tick_params(axis ='x', rotation =90)
    
    # Create legend & Show graphic
    plt.legend()
    plt.show()
    st.pyplot(f)


# Plot 3
if chart_visual_tweets=='Line Chart':
    # Plotting a line plot after changing it's width and height
    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(8)
    ls = ['dashed', 'solid']
    for i,lab in enumerate(label):
        plt.plot(count_of_data[i], label=label_name[i], linestyle=ls[i], marker='o')
    plt.xlabel('Time Series', fontweight='bold')
    plt.tick_params(axis ='x', rotation =90)
    # Create legend & Show graphic
    plt.legend()
    plt.show()
    st.pyplot(f)


################################### Time Series Analysis starts here
st.title("Time Series Analysis of Tweets")
chart_visual_time_series = st.sidebar.selectbox('Select Need/Availability Label for Time series distribution',('Need', 'Availability')) 
# y represemts the Need Label
# z represents the Availability Label
y = df['Need']
z = df['Availability']

if chart_visual_time_series=='Need':
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(y, marker='o', linewidth=0.5, label='Daily',ls='solid', c='red')
    ax.plot(y.resample('3D').mean(),marker='o', markersize=8, linestyle='dashed', label='Half-Weekly Mean Resample')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Date')
    ax.legend()
    st.pyplot(fig)

if chart_visual_time_series=="Availability":
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(z, marker='o', linewidth=0.5, label='Daily',ls='solid', c='red')
    ax.plot(z.resample('3D').mean(),marker='o', markersize=8, linestyle='dashed', label='Half-Weekly Mean Resample', color='blue')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Date')
    ax.legend()
    st.pyplot(fig)


################################### Seasonal Decomposition starts here
# The next step is to decompose the data to view more of the complexity behind the linear visualization. 
# A useful Python function called seasonal_decompose within the 'statsmodels' package can help us to decompose the data 
# into four different components:
# Observed
# Trended
# Seasonal
# Residual
st.title("Decompose the Data")
chart_visual_seasonal_decomposition = st.sidebar.selectbox('Select Need/Availability Label for Seasonal decomposition', 
                                    ('Need of resources', 'Availability of resources'))


def seasonal_decompose (x):
    decomposition_x = sm.tsa.seasonal_decompose(x, model='additive',extrapolate_trend='freq')
    fig_x = decomposition_x.plot()
    fig_x.set_size_inches(14,7)
    plt.show()
    st.pyplot(fig_x)
    
if chart_visual_seasonal_decomposition == "Need of resources":
    seasonal_decompose(y)
elif chart_visual_seasonal_decomposition == "Availability of resources":
    seasonal_decompose(z)
    
    
# Footer
footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

footer {visibility: hidden;}

.footer {
margin:0;
height:5px;
position:relative;
top:140px;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with <span style='color:red;'>❤</span> by <a style='text-align: center;' href="https://github.com/26aseem" target="_blank">Aseem Khullar</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)