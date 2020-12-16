<center><img src="tornado.jpg" width=820 height=600 /></center>

<h1><center><i> An Analysis of Climate Change and Natural Disasters</i></center></h1>

<center>Brenton Haliw</center>


<hr>
<h2><center> Introduction </center></h2>

As the Earth's climate has changed over the years, we often hear of the effects that this can bring. However, what exactly is affected? Does this mean that there will be more tropical cyclones and tornadoes each year, or will their intesity begin to shift? Would climate change affect the number of wildfires that occur in the world? If so, by how much? In order to answer these questions, I will be comparing and contrasting the numbers to see if there is a trend and what we can expect for the future of our planet if the Earth's climate continues to change.

To learn more about the different types of natural disasters before we begin, please feel free to click on the following links:

[What is a Hurricane, Typhoon, or Tropical Cyclone?](https://gpm.nasa.gov/education/articles/what-hurricane-typhoon-or-tropical-cyclone)


<hr>
<h2><center>Data Collection</center></h2>

#### Assets to import for data manipulation
In order to fully evaluate the link between climate change and natural disasters, we will import the following items. These packages will allow us to display statistical values, graphs, and a wide variety of ease of programming factors.


```python
!pip install folium

# These are our imports that we will use for the project
import re, requests, glob, folium, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn import linear_model
import numpy as np
from ipywidgets import *
import ipywidgets as widgets
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
import statsmodels.api as sm
```

    Collecting folium
      Downloading folium-0.11.0-py2.py3-none-any.whl (93 kB)
    [K     |████████████████████████████████| 93 kB 1.1 MB/s eta 0:00:01
    [?25hRequirement already satisfied: requests in /opt/conda/lib/python3.8/site-packages (from folium) (2.24.0)
    Collecting branca>=0.3.0
      Downloading branca-0.4.1-py3-none-any.whl (24 kB)
    Requirement already satisfied: jinja2>=2.9 in /opt/conda/lib/python3.8/site-packages (from folium) (2.11.2)
    Requirement already satisfied: numpy in /opt/conda/lib/python3.8/site-packages (from folium) (1.19.1)
    Requirement already satisfied: chardet<4,>=3.0.2 in /opt/conda/lib/python3.8/site-packages (from requests->folium) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /opt/conda/lib/python3.8/site-packages (from requests->folium) (2.10)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/lib/python3.8/site-packages (from requests->folium) (1.25.10)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.8/site-packages (from requests->folium) (2020.6.20)
    Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/lib/python3.8/site-packages (from jinja2>=2.9->folium) (1.1.1)
    Installing collected packages: branca, folium
    Successfully installed branca-0.4.1 folium-0.11.0


#### Collecting data for global temperature over the years
Our first order of business is to actually collect the data that we want to use to to make these comparisons. Therefore, a good place to start is to get the average global temperature for each year, and the number of different natural disasters that occur around the world.

By using the table from [NOAA](https://www.ncdc.noaa.gov/cag/global/time-series/globe/land_ocean/ytd/12/1880-2019), we can see what the global temperature anamoly each year is starting from 1880. This shows if the global temperature was above or below the estimated global temperature under normal conditions.


```python
# Reading the CSV file for the global temperature anamoly
temp_anamoly_df = pd.read_csv("files/temperature/data.csv")
temp_anamoly_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Global Land and Ocean Temperature Anomalies</th>
      <th>January-December</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Units: Degrees Celsius</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Base Period: 1901-2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Missing: -999</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Year</td>
      <td>Value</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1880</td>
      <td>-0.12</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>139</th>
      <td>2015</td>
      <td>0.93</td>
    </tr>
    <tr>
      <th>140</th>
      <td>2016</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>141</th>
      <td>2017</td>
      <td>0.91</td>
    </tr>
    <tr>
      <th>142</th>
      <td>2018</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>143</th>
      <td>2019</td>
      <td>0.95</td>
    </tr>
  </tbody>
</table>
<p>144 rows × 2 columns</p>
</div>



#### Collecting data about different types of natural disasters
Next, we will collect data concening tornadoes. The CSV files were downloaded from [NOAA](https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/). I saved all of the files to a folder and used a for loop to go through each CSV file and add it to a dataframe to be utilized later.


```python
# Creating the dataframes to store our information
storm_df = pd.DataFrame()
temp_csv = pd.DataFrame()

# The for loop will read each CSV file in the 'storm' folder and then add it to the storm dataframe.
for name in glob.glob("files/storm/*.csv"):
    # Creates a temporary dataframe containing the information related to the current storm data
    temp_csv = pd.read_csv(name)
    # Appending the temporary storm dataframe to the main dataframe
    storm_df = storm_df.append(temp_csv, ignore_index=True)
    
storm_df
```

    /opt/conda/lib/python3.8/site-packages/IPython/core/interactiveshell.py:3145: DtypeWarning: Columns (26,28) have mixed types.Specify dtype option on import or set low_memory=False.
      has_raised = await self.run_ast_nodes(code_ast.body, cell_name,
    /opt/conda/lib/python3.8/site-packages/IPython/core/interactiveshell.py:3145: DtypeWarning: Columns (28) have mixed types.Specify dtype option on import or set low_memory=False.
      has_raised = await self.run_ast_nodes(code_ast.body, cell_name,
    /opt/conda/lib/python3.8/site-packages/IPython/core/interactiveshell.py:3145: DtypeWarning: Columns (29,34,35,37) have mixed types.Specify dtype option on import or set low_memory=False.
      has_raised = await self.run_ast_nodes(code_ast.body, cell_name,





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BEGIN_YEARMONTH</th>
      <th>BEGIN_DAY</th>
      <th>BEGIN_TIME</th>
      <th>END_YEARMONTH</th>
      <th>END_DAY</th>
      <th>END_TIME</th>
      <th>EPISODE_ID</th>
      <th>EVENT_ID</th>
      <th>STATE</th>
      <th>STATE_FIPS</th>
      <th>...</th>
      <th>END_RANGE</th>
      <th>END_AZIMUTH</th>
      <th>END_LOCATION</th>
      <th>BEGIN_LAT</th>
      <th>BEGIN_LON</th>
      <th>END_LAT</th>
      <th>END_LON</th>
      <th>EPISODE_NARRATIVE</th>
      <th>EVENT_NARRATIVE</th>
      <th>DATA_SOURCE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>195004</td>
      <td>28</td>
      <td>1445</td>
      <td>195004</td>
      <td>28</td>
      <td>1445</td>
      <td>NaN</td>
      <td>10096222</td>
      <td>OKLAHOMA</td>
      <td>40.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>35.1200</td>
      <td>-99.2000</td>
      <td>35.1700</td>
      <td>-99.2000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>PUB</td>
    </tr>
    <tr>
      <th>1</th>
      <td>195004</td>
      <td>29</td>
      <td>1530</td>
      <td>195004</td>
      <td>29</td>
      <td>1530</td>
      <td>NaN</td>
      <td>10120412</td>
      <td>TEXAS</td>
      <td>48.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>31.9000</td>
      <td>-98.6000</td>
      <td>31.7300</td>
      <td>-98.6000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>PUB</td>
    </tr>
    <tr>
      <th>2</th>
      <td>195007</td>
      <td>5</td>
      <td>1800</td>
      <td>195007</td>
      <td>5</td>
      <td>1800</td>
      <td>NaN</td>
      <td>10104927</td>
      <td>PENNSYLVANIA</td>
      <td>42.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>40.5800</td>
      <td>-75.7000</td>
      <td>40.6500</td>
      <td>-75.4700</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>PUB</td>
    </tr>
    <tr>
      <th>3</th>
      <td>195007</td>
      <td>5</td>
      <td>1830</td>
      <td>195007</td>
      <td>5</td>
      <td>1830</td>
      <td>NaN</td>
      <td>10104928</td>
      <td>PENNSYLVANIA</td>
      <td>42.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>40.6000</td>
      <td>-76.7500</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>PUB</td>
    </tr>
    <tr>
      <th>4</th>
      <td>195007</td>
      <td>24</td>
      <td>1440</td>
      <td>195007</td>
      <td>24</td>
      <td>1440</td>
      <td>NaN</td>
      <td>10104929</td>
      <td>PENNSYLVANIA</td>
      <td>42.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.6300</td>
      <td>-79.6800</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>PUB</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1649644</th>
      <td>202006</td>
      <td>18</td>
      <td>600</td>
      <td>202006</td>
      <td>19</td>
      <td>600</td>
      <td>147668.0</td>
      <td>899187</td>
      <td>IOWA</td>
      <td>19.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>S</td>
      <td>HAMPTON</td>
      <td>42.7400</td>
      <td>-93.2000</td>
      <td>42.7400</td>
      <td>-93.2000</td>
      <td>An upper-level trough with associated surface ...</td>
      <td>Coop observer reported a 24 hour rainfall tota...</td>
      <td>CSV</td>
    </tr>
    <tr>
      <th>1649645</th>
      <td>202006</td>
      <td>18</td>
      <td>600</td>
      <td>202006</td>
      <td>19</td>
      <td>600</td>
      <td>147668.0</td>
      <td>899188</td>
      <td>IOWA</td>
      <td>19.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>N</td>
      <td>HAMPTON</td>
      <td>42.7600</td>
      <td>-93.2000</td>
      <td>42.7600</td>
      <td>-93.2000</td>
      <td>An upper-level trough with associated surface ...</td>
      <td>Coop observer reported a 24 hour rainfall tota...</td>
      <td>CSV</td>
    </tr>
    <tr>
      <th>1649646</th>
      <td>202006</td>
      <td>22</td>
      <td>2320</td>
      <td>202006</td>
      <td>24</td>
      <td>915</td>
      <td>148769.0</td>
      <td>899497</td>
      <td>IOWA</td>
      <td>19.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>E</td>
      <td>TRAER</td>
      <td>42.2000</td>
      <td>-92.4700</td>
      <td>42.2003</td>
      <td>-92.4637</td>
      <td>A surface low to the northwest of Iowa allowed...</td>
      <td>The department of transportation relayed a rep...</td>
      <td>CSV</td>
    </tr>
    <tr>
      <th>1649647</th>
      <td>202006</td>
      <td>23</td>
      <td>38</td>
      <td>202006</td>
      <td>24</td>
      <td>915</td>
      <td>148769.0</td>
      <td>899498</td>
      <td>IOWA</td>
      <td>19.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>E</td>
      <td>TRAER</td>
      <td>42.2000</td>
      <td>-92.4700</td>
      <td>42.2001</td>
      <td>-92.4647</td>
      <td>A surface low to the northwest of Iowa allowed...</td>
      <td>Iowa Department of Transportation and the Tama...</td>
      <td>CSV</td>
    </tr>
    <tr>
      <th>1649648</th>
      <td>202006</td>
      <td>23</td>
      <td>124</td>
      <td>202006</td>
      <td>24</td>
      <td>915</td>
      <td>148769.0</td>
      <td>899499</td>
      <td>IOWA</td>
      <td>19.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>SE</td>
      <td>IRVING</td>
      <td>41.9645</td>
      <td>-92.2974</td>
      <td>41.9056</td>
      <td>-92.2971</td>
      <td>A surface low to the northwest of Iowa allowed...</td>
      <td>Iowa Department of Transportation reports wate...</td>
      <td>CSV</td>
    </tr>
  </tbody>
</table>
<p>1649649 rows × 51 columns</p>
</div>



#### Collecting data about tropical cyclones over the years
Finally, I used [NOAA](https://www.nhc.noaa.gov/data/) to obtain data relating to tropical cyclones that have spanwed in the Pacfic Ocean and the Atlantic Ocean.


```python
atlantic_df = pd.read_csv("files/hurricane/hurdat2-1851-2019-052520.txt")
atlantic_df
```

    /opt/conda/lib/python3.8/site-packages/IPython/core/interactiveshell.py:3145: DtypeWarning: Columns (20) have mixed types.Specify dtype option on import or set low_memory=False.
      has_raised = await self.run_ast_nodes(code_ast.body, cell_name,





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th>AL011851</th>
      <th>UNNAMED</th>
      <th>14</th>
      <th>Unnamed: 3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">18510625</th>
      <th>0000</th>
      <th></th>
      <th>HU</th>
      <th>28.0N</th>
      <th>94.8W</th>
      <th>80.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0600</th>
      <th></th>
      <th>HU</th>
      <th>28.0N</th>
      <th>95.4W</th>
      <th>80.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1200</th>
      <th></th>
      <th>HU</th>
      <th>28.0N</th>
      <th>96.0W</th>
      <th>80.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1800</th>
      <th></th>
      <th>HU</th>
      <th>28.1N</th>
      <th>96.5W</th>
      <th>80.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2100</th>
      <th>L</th>
      <th>HU</th>
      <th>28.2N</th>
      <th>96.8W</th>
      <th>80.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">20191126</th>
      <th>1200</th>
      <th></th>
      <th>EX</th>
      <th>52.2N</th>
      <th>9.3W</th>
      <th>45.0</th>
      <th>970.0</th>
      <th>90.0</th>
      <th>240.0</th>
      <th>120.0</th>
      <th>90.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1800</th>
      <th></th>
      <th>EX</th>
      <th>52.2N</th>
      <th>8.9W</th>
      <th>40.0</th>
      <th>972.0</th>
      <th>90.0</th>
      <th>240.0</th>
      <th>90.0</th>
      <th>90.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">20191127</th>
      <th>0000</th>
      <th></th>
      <th>EX</th>
      <th>51.8N</th>
      <th>8.2W</th>
      <th>40.0</th>
      <th>974.0</th>
      <th>0.0</th>
      <th>210.0</th>
      <th>90.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0600</th>
      <th></th>
      <th>EX</th>
      <th>51.4N</th>
      <th>6.0W</th>
      <th>40.0</th>
      <th>976.0</th>
      <th>0.0</th>
      <th>180.0</th>
      <th>90.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1200</th>
      <th></th>
      <th>EX</th>
      <th>51.3N</th>
      <th>2.1W</th>
      <th>40.0</th>
      <th>980.0</th>
      <th>0.0</th>
      <th>120.0</th>
      <th>90.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>53732 rows × 4 columns</p>
</div>




```python
pacific_df = pd.read_csv("files/hurricane/hurdat2-nepac-1949-2019-042320.txt")
pacific_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th>EP011949</th>
      <th>UNNAMED</th>
      <th>7</th>
      <th>Unnamed: 3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">19490611</th>
      <th>0000</th>
      <th></th>
      <th>TS</th>
      <th>20.2N</th>
      <th>106.3W</th>
      <th>45.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0600</th>
      <th></th>
      <th>TS</th>
      <th>20.2N</th>
      <th>106.4W</th>
      <th>45.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1200</th>
      <th></th>
      <th>TS</th>
      <th>20.2N</th>
      <th>106.7W</th>
      <th>45.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1800</th>
      <th></th>
      <th>TS</th>
      <th>20.3N</th>
      <th>107.7W</th>
      <th>45.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19490612</th>
      <th>0000</th>
      <th></th>
      <th>TS</th>
      <th>20.4N</th>
      <th>108.6W</th>
      <th>45.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <th>-999.0</th>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">20191117</th>
      <th>0600</th>
      <th></th>
      <th>TD</th>
      <th>10.0N</th>
      <th>102.2W</th>
      <th>25.0</th>
      <th>1007.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1200</th>
      <th></th>
      <th>TD</th>
      <th>10.4N</th>
      <th>102.9W</th>
      <th>25.0</th>
      <th>1007.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1800</th>
      <th></th>
      <th>TD</th>
      <th>11.0N</th>
      <th>103.6W</th>
      <th>25.0</th>
      <th>1007.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">20191118</th>
      <th>0000</th>
      <th></th>
      <th>TD</th>
      <th>11.7N</th>
      <th>104.2W</th>
      <th>25.0</th>
      <th>1007.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0600</th>
      <th></th>
      <th>TD</th>
      <th>12.2N</th>
      <th>104.8W</th>
      <th>25.0</th>
      <th>1007.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <th>0.0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>30194 rows × 4 columns</p>
</div>



<hr>
<h2><center>Data Processing</center></h2>

#### Global Temperature DataFrame

Now that we have all of our required data, we need to be able to make sense of it. For example, our global temperature anamoly file that we read in was not formatted correctly, giving us a wonky dataframe with a few extra columns and a useless row. 

In order to make it usable later on in the project, I removed the first four rows of data, renamed the headers for the columns, and removed whitespace after the values in the Celsius column. The Year and Celsius columns were give their correct data types and the processing was complete for this dataframe.


```python
# Dropping unneeded rows due to formatting issues
temp_anamoly_df = temp_anamoly_df.drop([0, 1, 2, 3], axis=0)

# Renaming columns to their appropriate values
temp_anamoly_df = temp_anamoly_df.rename(columns={"Global Land and Ocean Temperature Anomalies": "YEAR", " January-December": "Celsius"})

# Changing column types to fit their intended types
temp_anamoly_df["YEAR"] = temp_anamoly_df["YEAR"].astype(int)

for item in temp_anamoly_df["Celsius"]:
    item = item.strip()

temp_anamoly_df["Celsius"] = temp_anamoly_df["Celsius"].astype(float)

# Fixing the index of the dataframe
temp_anamoly_df = temp_anamoly_df.reset_index()
temp_anamoly_df = temp_anamoly_df.drop("index", axis=1)
temp_anamoly_df.index += 1
temp_anamoly_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>Celsius</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1880</td>
      <td>-0.12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1881</td>
      <td>-0.09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1882</td>
      <td>-0.10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1883</td>
      <td>-0.18</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1884</td>
      <td>-0.27</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>136</th>
      <td>2015</td>
      <td>0.93</td>
    </tr>
    <tr>
      <th>137</th>
      <td>2016</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>138</th>
      <td>2017</td>
      <td>0.91</td>
    </tr>
    <tr>
      <th>139</th>
      <td>2018</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>140</th>
      <td>2019</td>
      <td>0.95</td>
    </tr>
  </tbody>
</table>
<p>140 rows × 2 columns</p>
</div>



#### Dissecting the Storm DataFrame
Looking through all one million rows of our storm dataframe, we see that there are a lot of entries that we don't really need. In order to highlight some of the more pressing natural disasters, we'll iterate through the rows of the storm dataframe and create separated dataframes for each natural disaster that we want to focus on (tornado, drought, flood, etc.)

By using a for loop, we iterate through each row and use an if statement to locate the type of natural disaster that we want to focus on. Once we find the natural disaster, we then add that row to a list to be stored for creating the specific dataframe.


```python
# Creating lists to store the rows for the applicable dataframes
tornado_row_list = []
drought_row_list = []
wildfire_row_list = []

# Iterrating through the rows of the storm dataframe and assigning the rows to the correct list
for index, row in storm_df.iterrows():
    if row["EVENT_TYPE"] == "Tornado":
        tornado_row_list.append(row)
    
    if row["EVENT_TYPE"] == "Drought":
        drought_row_list.append(row)

    if row["EVENT_TYPE"] == "Wildfire":
        wildfire_row_list.append(row)

columns = ["YEAR", "BEGIN_DATE_TIME", "STATE", "DAMAGE_PROPERTY"]
tornado_columns = columns + ["TOR_F_SCALE", "TOR_LENGTH", "TOR_WIDTH", "BEGIN_LAT", "BEGIN_LON"]
```

Now that we have our information that we want to use, we can go ahead and create our dataframes for the natural disasters that we chose.

#### Tornado DataFrame Creation
To create the tornado dataframe, we used the list of tornado columns that we created above, and the list containing the rows of data to populate our new dataframe with. Once we did this, we cleaned up the indices of the dataframe to allow them to stat from 1 to n.


```python
# Creating a temporary dataframe to store all the information collected from original storm dataframe
temp_df = pd.DataFrame()
temp_df = temp_df.append(tornado_row_list)

# Creating the tornado dataframe with only the columns that we want
tornado_df = temp_df.filter(tornado_columns)

# Cleaning up the indices of the tornado dataframe
tornado_df = tornado_df.reset_index(drop=True)
tornado_df.index += 1
tornado_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>BEGIN_DATE_TIME</th>
      <th>STATE</th>
      <th>DAMAGE_PROPERTY</th>
      <th>TOR_F_SCALE</th>
      <th>TOR_LENGTH</th>
      <th>TOR_WIDTH</th>
      <th>BEGIN_LAT</th>
      <th>BEGIN_LON</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1950</td>
      <td>28-APR-50 14:45:00</td>
      <td>OKLAHOMA</td>
      <td>250K</td>
      <td>F3</td>
      <td>3.40</td>
      <td>400.0</td>
      <td>35.1200</td>
      <td>-99.2000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1950</td>
      <td>29-APR-50 15:30:00</td>
      <td>TEXAS</td>
      <td>25K</td>
      <td>F1</td>
      <td>11.50</td>
      <td>200.0</td>
      <td>31.9000</td>
      <td>-98.6000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1950</td>
      <td>05-JUL-50 18:00:00</td>
      <td>PENNSYLVANIA</td>
      <td>25K</td>
      <td>F2</td>
      <td>12.90</td>
      <td>33.0</td>
      <td>40.5800</td>
      <td>-75.7000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1950</td>
      <td>05-JUL-50 18:30:00</td>
      <td>PENNSYLVANIA</td>
      <td>2.5K</td>
      <td>F2</td>
      <td>0.00</td>
      <td>13.0</td>
      <td>40.6000</td>
      <td>-76.7500</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1950</td>
      <td>24-JUL-50 14:40:00</td>
      <td>PENNSYLVANIA</td>
      <td>2.5K</td>
      <td>F0</td>
      <td>0.00</td>
      <td>33.0</td>
      <td>41.6300</td>
      <td>-79.6800</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>71969</th>
      <td>2020</td>
      <td>04-JUN-20 01:59:00</td>
      <td>MISSOURI</td>
      <td>NaN</td>
      <td>EF0</td>
      <td>0.63</td>
      <td>40.0</td>
      <td>38.4719</td>
      <td>-94.3316</td>
    </tr>
    <tr>
      <th>71970</th>
      <td>2020</td>
      <td>28-JUN-20 19:54:00</td>
      <td>MINNESOTA</td>
      <td>2.00K</td>
      <td>EF0</td>
      <td>0.61</td>
      <td>20.0</td>
      <td>44.0318</td>
      <td>-91.7238</td>
    </tr>
    <tr>
      <th>71971</th>
      <td>2020</td>
      <td>10-JUN-20 17:18:00</td>
      <td>OHIO</td>
      <td>50.00K</td>
      <td>EF0</td>
      <td>5.90</td>
      <td>300.0</td>
      <td>39.3786</td>
      <td>-83.0791</td>
    </tr>
    <tr>
      <th>71972</th>
      <td>2020</td>
      <td>07-JUN-20 17:41:00</td>
      <td>FLORIDA</td>
      <td>0.00K</td>
      <td>EF0</td>
      <td>0.50</td>
      <td>20.0</td>
      <td>28.7914</td>
      <td>-81.7040</td>
    </tr>
    <tr>
      <th>71973</th>
      <td>2020</td>
      <td>02-JUN-20 14:02:00</td>
      <td>KANSAS</td>
      <td>0.00K</td>
      <td>EFU</td>
      <td>0.57</td>
      <td>30.0</td>
      <td>38.8031</td>
      <td>-101.7521</td>
    </tr>
  </tbody>
</table>
<p>71973 rows × 9 columns</p>
</div>



#### Drought DataFrame Creation


```python
# Creating a temporary dataframe to store all the information collected from original storm dataframe
temp_df = pd.DataFrame()
temp_df = temp_df.append(drought_row_list)

# Creating the drought dataframe with only the columns that we want
drought_df = temp_df.filter(columns)

# Cleaning up the indices of the drought dataframe
drought_df = drought_df.reset_index(drop=True)
drought_df.index += 1
drought_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>BEGIN_DATE_TIME</th>
      <th>STATE</th>
      <th>DAMAGE_PROPERTY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1996</td>
      <td>01-MAY-96 00:00:00</td>
      <td>TEXAS</td>
      <td>.1M</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1996</td>
      <td>01-MAY-96 00:00:00</td>
      <td>TEXAS</td>
      <td>.1M</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1996</td>
      <td>01-MAY-96 00:00:00</td>
      <td>TEXAS</td>
      <td>.1M</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1996</td>
      <td>01-MAY-96 00:00:00</td>
      <td>TEXAS</td>
      <td>.1M</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1996</td>
      <td>01-MAY-96 00:00:00</td>
      <td>TEXAS</td>
      <td>.1M</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>56458</th>
      <td>2020</td>
      <td>01-JUN-20 00:00:00</td>
      <td>KANSAS</td>
      <td>0.00K</td>
    </tr>
    <tr>
      <th>56459</th>
      <td>2020</td>
      <td>01-JUN-20 00:00:00</td>
      <td>OKLAHOMA</td>
      <td>0.00K</td>
    </tr>
    <tr>
      <th>56460</th>
      <td>2020</td>
      <td>01-JUN-20 00:00:00</td>
      <td>KANSAS</td>
      <td>0.00K</td>
    </tr>
    <tr>
      <th>56461</th>
      <td>2020</td>
      <td>01-JUN-20 00:00:00</td>
      <td>TEXAS</td>
      <td>0.00K</td>
    </tr>
    <tr>
      <th>56462</th>
      <td>2020</td>
      <td>01-JUN-20 00:00:00</td>
      <td>TEXAS</td>
      <td>0.00K</td>
    </tr>
  </tbody>
</table>
<p>56462 rows × 4 columns</p>
</div>



#### Wildfire DataFrame Creation


```python
# Creating a temporary dataframe to store all the information collected from original storm dataframe
temp_df = pd.DataFrame()
temp_df = temp_df.append(wildfire_row_list)

# Creating the wildfire dataframe with only the columns that we want
wildfire_df = temp_df.filter(columns)

# Cleaning up the indices of the wildfire dataframe
wildfire_df = wildfire_df.reset_index(drop=True)
wildfire_df.index += 1
wildfire_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>BEGIN_DATE_TIME</th>
      <th>STATE</th>
      <th>DAMAGE_PROPERTY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1996</td>
      <td>24-FEB-96 10:45:00</td>
      <td>KANSAS</td>
      <td>.25M</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1996</td>
      <td>04-MAR-96 12:00:00</td>
      <td>FLORIDA</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1996</td>
      <td>15-FEB-96 10:00:00</td>
      <td>TEXAS</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1996</td>
      <td>22-FEB-96 08:00:00</td>
      <td>TEXAS</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1996</td>
      <td>02-MAR-96 12:00:00</td>
      <td>TEXAS</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7367</th>
      <td>2020</td>
      <td>28-JUN-20 12:36:00</td>
      <td>COLORADO</td>
      <td>0.00K</td>
    </tr>
    <tr>
      <th>7368</th>
      <td>2020</td>
      <td>15-JUN-20 13:30:00</td>
      <td>COLORADO</td>
      <td>0.00K</td>
    </tr>
    <tr>
      <th>7369</th>
      <td>2020</td>
      <td>05-JUN-20 15:49:00</td>
      <td>COLORADO</td>
      <td>0.00K</td>
    </tr>
    <tr>
      <th>7370</th>
      <td>2020</td>
      <td>04-JUN-20 17:00:00</td>
      <td>CALIFORNIA</td>
      <td>0.00K</td>
    </tr>
    <tr>
      <th>7371</th>
      <td>2020</td>
      <td>03-JUN-20 10:55:00</td>
      <td>CALIFORNIA</td>
      <td>0.00K</td>
    </tr>
  </tbody>
</table>
<p>7371 rows × 4 columns</p>
</div>



#### Tropical Cyclone DataFrame Creation
Unfortunately, the files containing the information for stropical cyclone were not as clearly laid out as the information for the storm related information. Therefore, we have to completely redo everything by opening the file up through Python and manipulate it in a way that is more legible for Pandas to interpret it.

First, I'll create a function that will easily allow us to manipulate the data within the tropical cyclone files to a more readable format for Pandas. Once that is accomplished, I will then create a dataframe by using the read_csv() function on the new text file and then tidy it up. Then, I'll return the brand new dataframe for the specific ocean containing the tropical storms. 


```python
# Creating a function to create tropical cyclone dataframes so we don't duplicate code
def create_cyclone_df(read_file_path, write_file_path):
    # Open files for reading and writing
    file = open(read_file_path, "r")
    new_file = open(write_file_path, "w")
    
    # Temporary variables to use within the for loop
    cyclone_designator = ""
    cyclone_name = ""
        
    # Looping through the lines in the file. New Tropical Cyclones are denoted by AL followed by cylone number and YYYY
    # Therefore, we will identify these lines first, and then move this information to each line for the tropical cyclone.
    #
    # Example: AL011851, UNNAMED
    #          18510625, 0000, .....
    #          18510625, 0200, .....
    #
    # Will be:  AL011851, UNNAMED, 18510625, 0000, ....
    #           AL011851, UNNAMED, 18510625, 0200, ....
    for line in file:
        # Check if the line starts with two characters followed by the cyclone number and date
        if re.search("^[A-Z]{2}\d+", line):
            # Split the line into an array of strings at each comma
            split = line.split(",")
            # Assign our variables the values in the array and remove any whitespace
            cyclone_designator = split[0].strip()
            cyclone_name = split[1].strip()
    
        # Writing a new line to our file
        else:
            new_line = cyclone_designator + ", " + cyclone_name + ", " + line
            new_file.write(new_line)
    
    # Closing the files since we don't need them anymore
    new_file.close()
    file.close()
    
    # Creating the cyclone dataframe from the new file
    column_names = ["Designator", "Name", "Date_Group", "Time", "Record_Identifier", "Status", "Lat", "Long", "Max_Wind_(mph)", "Central_Pressure_(mbar)"]
    
    # Reading our new file and creating the dataframe
    cyclone_df = pd.read_csv(write_file_path, header=None)
    
    # Tidy up the data within the dataframe
    cyclone_df = cyclone_df.drop(list(cyclone_df)[10:], axis=1)
    cyclone_df.columns = column_names
    cyclone_df = cyclone_df.replace(-999, np.NaN)
    cyclone_df.index += 1
    
    # Create a Year column to easily group cyclones up
    year_column = []
    for index, row in cyclone_df.iterrows():
        year = str(row["Date_Group"]).strip()
        year_column.append(year[0:4])
    cyclone_df["YEAR"] = year_column
    cyclone_df["YEAR"] = cyclone_df["YEAR"].astype(int)
    
    return cyclone_df
```

Now that we have our function in place, we can use it to create our Atlantic Ocean tropical cyclone dataframe.


```python
# Creating dataframe for Tropical Cyclones in the Atlantic Ocean using the function from above
read_file = "files/hurricane/hurdat2-1851-2019-052520.txt"
write_file = "files/hurricane/atlantic_hurricane_data.txt"

atlantic_cyclone_df = create_cyclone_df(read_file, write_file)
atlantic_cyclone_df
```

    /opt/conda/lib/python3.8/site-packages/IPython/core/interactiveshell.py:3337: DtypeWarning: Columns (22) have mixed types.Specify dtype option on import or set low_memory=False.
      if (await self.run_code(code, result,  async_=asy)):





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Designator</th>
      <th>Name</th>
      <th>Date_Group</th>
      <th>Time</th>
      <th>Record_Identifier</th>
      <th>Status</th>
      <th>Lat</th>
      <th>Long</th>
      <th>Max_Wind_(mph)</th>
      <th>Central_Pressure_(mbar)</th>
      <th>YEAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>AL011851</td>
      <td>UNNAMED</td>
      <td>18510625</td>
      <td>0</td>
      <td></td>
      <td>HU</td>
      <td>28.0N</td>
      <td>94.8W</td>
      <td>80</td>
      <td>NaN</td>
      <td>1851</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AL011851</td>
      <td>UNNAMED</td>
      <td>18510625</td>
      <td>600</td>
      <td></td>
      <td>HU</td>
      <td>28.0N</td>
      <td>95.4W</td>
      <td>80</td>
      <td>NaN</td>
      <td>1851</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AL011851</td>
      <td>UNNAMED</td>
      <td>18510625</td>
      <td>1200</td>
      <td></td>
      <td>HU</td>
      <td>28.0N</td>
      <td>96.0W</td>
      <td>80</td>
      <td>NaN</td>
      <td>1851</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AL011851</td>
      <td>UNNAMED</td>
      <td>18510625</td>
      <td>1800</td>
      <td></td>
      <td>HU</td>
      <td>28.1N</td>
      <td>96.5W</td>
      <td>80</td>
      <td>NaN</td>
      <td>1851</td>
    </tr>
    <tr>
      <th>5</th>
      <td>AL011851</td>
      <td>UNNAMED</td>
      <td>18510625</td>
      <td>2100</td>
      <td>L</td>
      <td>HU</td>
      <td>28.2N</td>
      <td>96.8W</td>
      <td>80</td>
      <td>NaN</td>
      <td>1851</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>51836</th>
      <td>AL202019</td>
      <td>SEBASTIEN</td>
      <td>20191126</td>
      <td>1200</td>
      <td></td>
      <td>EX</td>
      <td>52.2N</td>
      <td>9.3W</td>
      <td>45</td>
      <td>970.0</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>51837</th>
      <td>AL202019</td>
      <td>SEBASTIEN</td>
      <td>20191126</td>
      <td>1800</td>
      <td></td>
      <td>EX</td>
      <td>52.2N</td>
      <td>8.9W</td>
      <td>40</td>
      <td>972.0</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>51838</th>
      <td>AL202019</td>
      <td>SEBASTIEN</td>
      <td>20191127</td>
      <td>0</td>
      <td></td>
      <td>EX</td>
      <td>51.8N</td>
      <td>8.2W</td>
      <td>40</td>
      <td>974.0</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>51839</th>
      <td>AL202019</td>
      <td>SEBASTIEN</td>
      <td>20191127</td>
      <td>600</td>
      <td></td>
      <td>EX</td>
      <td>51.4N</td>
      <td>6.0W</td>
      <td>40</td>
      <td>976.0</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>51840</th>
      <td>AL202019</td>
      <td>SEBASTIEN</td>
      <td>20191127</td>
      <td>1200</td>
      <td></td>
      <td>EX</td>
      <td>51.3N</td>
      <td>2.1W</td>
      <td>40</td>
      <td>980.0</td>
      <td>2019</td>
    </tr>
  </tbody>
</table>
<p>51840 rows × 11 columns</p>
</div>



Now that we have our Atlantic Ocean tropical cyclone dataframe, we can do the same to get our Pacific Ocean tropical cyclone dataframe.


```python
# Creating dataframe for tropical cyclones in the Pacific Ocean using the function from above
read_file = "files/hurricane/hurdat2-nepac-1949-2019-042320.txt"
write_file = "files/hurricane/pacific_hurricane_data.txt"

pacific_cyclone_df = create_cyclone_df(read_file, write_file)
pacific_cyclone_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Designator</th>
      <th>Name</th>
      <th>Date_Group</th>
      <th>Time</th>
      <th>Record_Identifier</th>
      <th>Status</th>
      <th>Lat</th>
      <th>Long</th>
      <th>Max_Wind_(mph)</th>
      <th>Central_Pressure_(mbar)</th>
      <th>YEAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>EP011949</td>
      <td>UNNAMED</td>
      <td>19490611</td>
      <td>0</td>
      <td></td>
      <td>TS</td>
      <td>20.2N</td>
      <td>106.3W</td>
      <td>45</td>
      <td>NaN</td>
      <td>1949</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EP011949</td>
      <td>UNNAMED</td>
      <td>19490611</td>
      <td>600</td>
      <td></td>
      <td>TS</td>
      <td>20.2N</td>
      <td>106.4W</td>
      <td>45</td>
      <td>NaN</td>
      <td>1949</td>
    </tr>
    <tr>
      <th>3</th>
      <td>EP011949</td>
      <td>UNNAMED</td>
      <td>19490611</td>
      <td>1200</td>
      <td></td>
      <td>TS</td>
      <td>20.2N</td>
      <td>106.7W</td>
      <td>45</td>
      <td>NaN</td>
      <td>1949</td>
    </tr>
    <tr>
      <th>4</th>
      <td>EP011949</td>
      <td>UNNAMED</td>
      <td>19490611</td>
      <td>1800</td>
      <td></td>
      <td>TS</td>
      <td>20.3N</td>
      <td>107.7W</td>
      <td>45</td>
      <td>NaN</td>
      <td>1949</td>
    </tr>
    <tr>
      <th>5</th>
      <td>EP011949</td>
      <td>UNNAMED</td>
      <td>19490612</td>
      <td>0</td>
      <td></td>
      <td>TS</td>
      <td>20.4N</td>
      <td>108.6W</td>
      <td>45</td>
      <td>NaN</td>
      <td>1949</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>29044</th>
      <td>EP212019</td>
      <td>TWENTYONE</td>
      <td>20191117</td>
      <td>600</td>
      <td></td>
      <td>TD</td>
      <td>10.0N</td>
      <td>102.2W</td>
      <td>25</td>
      <td>1007.0</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>29045</th>
      <td>EP212019</td>
      <td>TWENTYONE</td>
      <td>20191117</td>
      <td>1200</td>
      <td></td>
      <td>TD</td>
      <td>10.4N</td>
      <td>102.9W</td>
      <td>25</td>
      <td>1007.0</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>29046</th>
      <td>EP212019</td>
      <td>TWENTYONE</td>
      <td>20191117</td>
      <td>1800</td>
      <td></td>
      <td>TD</td>
      <td>11.0N</td>
      <td>103.6W</td>
      <td>25</td>
      <td>1007.0</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>29047</th>
      <td>EP212019</td>
      <td>TWENTYONE</td>
      <td>20191118</td>
      <td>0</td>
      <td></td>
      <td>TD</td>
      <td>11.7N</td>
      <td>104.2W</td>
      <td>25</td>
      <td>1007.0</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>29048</th>
      <td>EP212019</td>
      <td>TWENTYONE</td>
      <td>20191118</td>
      <td>600</td>
      <td></td>
      <td>TD</td>
      <td>12.2N</td>
      <td>104.8W</td>
      <td>25</td>
      <td>1007.0</td>
      <td>2019</td>
    </tr>
  </tbody>
</table>
<p>29048 rows × 11 columns</p>
</div>



<hr>
<h2><center>Vizualization</center></h2>

The purpose of these graphs to just to get a feel for our data and to see how it looks for certain types of combinations. Once we get the graphs that we like, we will then use them together in our analysis phase in order to come to a conclusion on if the graphs correlate to each other in any way. In the next few sections, we will further break down the dataframes that we have created about in order to get ready to analyze them.

#### Global Temperature Anamoly Graph
The following graph shows the global temperature anamoly over the recorded years. We can see that that temperature anamoly has been rising steadily over the years.


```python
temp_plot = temp_anamoly_df.plot(x="YEAR", y="Celsius", title="Global Temperature Anamoly Over Time", figsize=(15,10))
temp_plot.set_xlabel("Year")
temp_plot.set_ylabel("Degrees (Celsius)")
```




    Text(0, 0.5, 'Degrees (Celsius)')




    
![png](output_35_1.png)
    


#### Number of Tornadoes in the United States Graph
The following graph shows the number of tornadoes that occured in the United States over a period of time. It appears that over time the number of tornadoes has been increasing. An interesting observation to make note of is just how few tornadoes were recorded from 1950 to about 1960. According to NOAA, about 1000 tornadoes occur annually in the United States. We'll revisit this topic in our analysis phase of the project.

Also to note, the United States changed from the Fujita Scale to the Enhanced Fujita Scale in 2007 to more accurately assess tornado strength and damage output. The wind speed for each scale is still somewhat close to each other, however.


```python
tornado_group = tornado_df.groupby("YEAR")
tornado_plot = tornado_group.size().plot(title="Number of Tornadoes Over Time", figsize=(15,10))
tornado_plot.set_xlabel("Year")
tornado_plot.set_ylabel("Number of Tornadoes")
```




    Text(0, 0.5, 'Number of Tornadoes')




    
![png](output_37_1.png)
    



```python
tornado_intensity_group = tornado_df.groupby(["YEAR", "TOR_F_SCALE"])
intensity_df = tornado_intensity_group.size().reset_index(name="counts")
plot = intensity_df.groupby(["YEAR", "TOR_F_SCALE"])["counts"].sum().unstack("TOR_F_SCALE").plot.bar(stacked=True, figsize=(15,10))
plot.set_title("Number of Tornadoes and Intensity over Time")
plot.set_xlabel("Year")
plot.set_ylabel("Number of Tornadoes")
```




    Text(0, 0.5, 'Number of Tornadoes')




    
![png](output_38_1.png)
    



```python
tornado_intensity_group = tornado_df.groupby(["YEAR", "TOR_F_SCALE"])
intensity_df = tornado_intensity_group.size().reset_index(name="counts")
plot = intensity_df.groupby(["YEAR", "TOR_F_SCALE"])["counts"].sum().unstack("TOR_F_SCALE").plot(figsize=(16,10))
plot.set_title("Number of Tornadoes and Intensity over Time")
plot.set_xlabel("Year")
plot.set_ylabel("Number of Tornadoes")
```




    Text(0, 0.5, 'Number of Tornadoes')




    
![png](output_39_1.png)
    


#### Number of Wildfires in the United States Graph
The following graph shows the number of wildfires that occured in the United States over a period of time.


```python
wildfire_group = wildfire_df.groupby("YEAR")
wildfire_plot = wildfire_group.size().plot(title="Number of Wildfires Over Time", figsize=(15,10))
wildfire_plot.set_xlabel("Year")
wildfire_plot.set_ylabel("Number of Wildfires")
```




    Text(0, 0.5, 'Number of Wildfires')




    
![png](output_41_1.png)
    


#### Number of Droughts in the United States Graph
The following graph shows the number of droughts that occured in the United states over a period of time.


```python
drought_group = drought_df.groupby("YEAR")
drought_plot = drought_group.size().plot(title="Number of Droughts Over Time", figsize=(15,10))
drought_plot.set_xlabel("Year")
drought_plot.set_ylabel("Number of Droughts")
```




    Text(0, 0.5, 'Number of Droughts')




    
![png](output_43_1.png)
    


#### Number of Tropical Cyclones in the Atlantic Ocean and the Pacific Ocean


```python
pacific_group = pacific_cyclone_df.groupby("YEAR")
pacific_plot = pacific_group.size().plot(title="Number of Tropical Cyclones in the Pacific Ocean per Year", figsize=(15,10))
pacific_plot.set_xlabel("Year")
pacific_plot.set_ylabel("Number of Tropical Cyclones")
```




    Text(0, 0.5, 'Number of Tropical Cyclones')




    
![png](output_45_1.png)
    



```python
atlantic_group = atlantic_cyclone_df.groupby("YEAR")
atlantic_plot = atlantic_group.size().plot(title="Number of Tropical Cyclones in the Atlantic Ocean per Year", figsize=(15,10))
atlantic_plot.set_xlabel("Year")
atlantic_plot.set_ylabel("Number of Tropical Cyclones")
```




    Text(0, 0.5, 'Number of Tropical Cyclones')




    
![png](output_46_1.png)
    


<hr>
<h2><center>Analysis and Hypothesis Testing</center></h2>

Now that we have some numerical data to look at, it's finally time to put it all together to see if there is an effect on natural disasters and the rising global temperature. To accomplish this, we will use linear regression in order to view the correlation.

For the purposes of this project, we will assume the hypothesis that with the increase in global temperature, the amount of natural disasters will increase.

We will used Ordianary Least Squares (OLS) regression in order to determine if the natural disasters and the rising global temperature are in any way related. If you wish to learn more about OLS regression with Python, feel free to visit [this website.](https://medium.com/@jyotiyadav99111/statistics-how-should-i-interpret-results-of-ols-3bde1ebeec01)

#### Global Temperature and Number of Tornadoes
We'll merge the tornado dataframe and the global temperature anamoly dataframe together, and then create another table based on the grouping of the years.


```python
# Grouping together the tornado dataframe by year and getting the number of tornadoes each year
group = tornado_df.groupby("YEAR").size().reset_index(name="NUM_TORNADO")

# Merging the tornado dataframe and the global temp anamomoly dataframe
group = pd.merge(group, temp_anamoly_df, on="YEAR")
group.index += 1
group
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>NUM_TORNADO</th>
      <th>Celsius</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1950</td>
      <td>223</td>
      <td>-0.15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1951</td>
      <td>269</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1952</td>
      <td>272</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1953</td>
      <td>492</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1954</td>
      <td>609</td>
      <td>-0.10</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>66</th>
      <td>2015</td>
      <td>1320</td>
      <td>0.93</td>
    </tr>
    <tr>
      <th>67</th>
      <td>2016</td>
      <td>1079</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>68</th>
      <td>2017</td>
      <td>1647</td>
      <td>0.91</td>
    </tr>
    <tr>
      <th>69</th>
      <td>2018</td>
      <td>1254</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>70</th>
      <td>2019</td>
      <td>1728</td>
      <td>0.95</td>
    </tr>
  </tbody>
</table>
<p>70 rows × 3 columns</p>
</div>




```python
# Dependent variable is the number of tornadoes and the independent variable is degrees Celsius.
test = smf.ols(formula="NUM_TORNADO ~ Celsius", data=group).fit()
test.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>NUM_TORNADO</td>   <th>  R-squared:         </th> <td>   0.486</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.478</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   64.22</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 16 Dec 2020</td> <th>  Prob (F-statistic):</th> <td>2.07e-11</td>
</tr>
<tr>
  <th>Time:</th>                 <td>12:36:35</td>     <th>  Log-Likelihood:    </th> <td> -492.66</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    70</td>      <th>  AIC:               </th> <td>   989.3</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    68</td>      <th>  BIC:               </th> <td>   993.8</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>  739.4950</td> <td>   47.715</td> <td>   15.498</td> <td> 0.000</td> <td>  644.280</td> <td>  834.710</td>
</tr>
<tr>
  <th>Celsius</th>   <td>  878.6269</td> <td>  109.636</td> <td>    8.014</td> <td> 0.000</td> <td>  659.851</td> <td> 1097.403</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 6.224</td> <th>  Durbin-Watson:     </th> <td>   1.514</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.045</td> <th>  Jarque-Bera (JB):  </th> <td>   5.399</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.591</td> <th>  Prob(JB):          </th> <td>  0.0672</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.674</td> <th>  Cond. No.          </th> <td>    3.63</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
fig = sm.graphics.plot_partregress_grid(test)
fig.tight_layout(pad=1.0)
```


    
![png](output_52_0.png)
    



```python
plot = group.plot.scatter(x="Celsius", y="NUM_TORNADO", figsize=(15,10))
reg = LinearRegression().fit(group["Celsius"].values.reshape(-1,1), group["NUM_TORNADO"])
plt.plot(group["Celsius"].values.reshape(-1,1), reg.predict(group["Celsius"].values.reshape(-1,1)))
plot.set_xlabel("Degrees (Celsius)")
plot.set_ylabel("Number of Tornadoes")
plot.set_title("Number of Tornadoes For Each Change in Celsius")
```




    Text(0.5, 1.0, 'Number of Tornadoes For Each Change in Celsius')




    
![png](output_53_1.png)
    


From the OLS Regression Results, we see that our Prob (F-statistic) is 2.07e-11, which is an incredibly small number. This means with the data supplied, there is a correlation between the number of tornadoes and the increasing global temperature. However! There is some bias that is also contained within this data, especially within the first 10 - 20 years. 

As stated earlier in the Visualization phase, NOAA estimates that on average there are about 1000 tornadoes in the United States each year. However, for the first 10 - 20 years of the graph, we see that there are fewer than 1000, with the early-to-mid 1950s only showing a few hundred. Does this mean that there were only 223 tornadoes in the United States in 1950? Probably not. Something that has also evolved during this time has been weather radar and predictions models, which can accurately show if a tornado has been produced by a thunder storm. So, maybe, the tornadoes that have been reported within the last 20 years are much more accurate than the tornadoes that were reported within the first 20 years of the data. So, just as a test, lets go ahead and drop 1950 - 1970 and see if that has any effect on our P-Number.


```python
# Remove the first 20 indices of the tornado dataframe used above
temp = group.drop(group.index[0:20], axis=0)
temp
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>NUM_TORNADO</th>
      <th>Celsius</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>1970</td>
      <td>700</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1971</td>
      <td>963</td>
      <td>-0.06</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1972</td>
      <td>778</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1973</td>
      <td>1198</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1974</td>
      <td>1120</td>
      <td>-0.06</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1975</td>
      <td>962</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1976</td>
      <td>935</td>
      <td>-0.07</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1977</td>
      <td>922</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1978</td>
      <td>875</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1979</td>
      <td>918</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1980</td>
      <td>972</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1981</td>
      <td>830</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>33</th>
      <td>1982</td>
      <td>1180</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>34</th>
      <td>1983</td>
      <td>995</td>
      <td>0.36</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1984</td>
      <td>1020</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1985</td>
      <td>773</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1986</td>
      <td>849</td>
      <td>0.24</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1987</td>
      <td>695</td>
      <td>0.39</td>
    </tr>
    <tr>
      <th>39</th>
      <td>1988</td>
      <td>773</td>
      <td>0.40</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1989</td>
      <td>921</td>
      <td>0.30</td>
    </tr>
    <tr>
      <th>41</th>
      <td>1990</td>
      <td>1264</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>42</th>
      <td>1991</td>
      <td>1208</td>
      <td>0.39</td>
    </tr>
    <tr>
      <th>43</th>
      <td>1992</td>
      <td>1404</td>
      <td>0.24</td>
    </tr>
    <tr>
      <th>44</th>
      <td>1993</td>
      <td>616</td>
      <td>0.29</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1994</td>
      <td>947</td>
      <td>0.35</td>
    </tr>
    <tr>
      <th>46</th>
      <td>1995</td>
      <td>1217</td>
      <td>0.47</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1996</td>
      <td>1267</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>48</th>
      <td>1997</td>
      <td>1180</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>49</th>
      <td>1998</td>
      <td>1529</td>
      <td>0.65</td>
    </tr>
    <tr>
      <th>50</th>
      <td>1999</td>
      <td>1520</td>
      <td>0.44</td>
    </tr>
    <tr>
      <th>51</th>
      <td>2000</td>
      <td>1169</td>
      <td>0.43</td>
    </tr>
    <tr>
      <th>52</th>
      <td>2001</td>
      <td>1351</td>
      <td>0.57</td>
    </tr>
    <tr>
      <th>53</th>
      <td>2002</td>
      <td>1040</td>
      <td>0.62</td>
    </tr>
    <tr>
      <th>54</th>
      <td>2003</td>
      <td>1535</td>
      <td>0.64</td>
    </tr>
    <tr>
      <th>55</th>
      <td>2004</td>
      <td>1947</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>56</th>
      <td>2005</td>
      <td>1343</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>57</th>
      <td>2006</td>
      <td>1263</td>
      <td>0.64</td>
    </tr>
    <tr>
      <th>58</th>
      <td>2007</td>
      <td>1241</td>
      <td>0.62</td>
    </tr>
    <tr>
      <th>59</th>
      <td>2008</td>
      <td>1954</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>60</th>
      <td>2009</td>
      <td>1273</td>
      <td>0.65</td>
    </tr>
    <tr>
      <th>61</th>
      <td>2010</td>
      <td>1449</td>
      <td>0.73</td>
    </tr>
    <tr>
      <th>62</th>
      <td>2011</td>
      <td>2074</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>63</th>
      <td>2012</td>
      <td>1058</td>
      <td>0.64</td>
    </tr>
    <tr>
      <th>64</th>
      <td>2013</td>
      <td>1053</td>
      <td>0.68</td>
    </tr>
    <tr>
      <th>65</th>
      <td>2014</td>
      <td>1055</td>
      <td>0.74</td>
    </tr>
    <tr>
      <th>66</th>
      <td>2015</td>
      <td>1320</td>
      <td>0.93</td>
    </tr>
    <tr>
      <th>67</th>
      <td>2016</td>
      <td>1079</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>68</th>
      <td>2017</td>
      <td>1647</td>
      <td>0.91</td>
    </tr>
    <tr>
      <th>69</th>
      <td>2018</td>
      <td>1254</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>70</th>
      <td>2019</td>
      <td>1728</td>
      <td>0.95</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Dependent variable is the number of tornadoes and the independent variable is degrees Celsius.
new_test = smf.ols(formula="NUM_TORNADO ~ Celsius", data=temp).fit()
new_test.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>NUM_TORNADO</td>   <th>  R-squared:         </th> <td>   0.294</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.280</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   20.02</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 16 Dec 2020</td> <th>  Prob (F-statistic):</th> <td>4.71e-05</td>
</tr>
<tr>
  <th>Time:</th>                 <td>12:36:35</td>     <th>  Log-Likelihood:    </th> <td> -351.50</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    50</td>      <th>  AIC:               </th> <td>   707.0</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    48</td>      <th>  BIC:               </th> <td>   710.8</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>  887.7090</td> <td>   73.906</td> <td>   12.011</td> <td> 0.000</td> <td>  739.111</td> <td> 1036.307</td>
</tr>
<tr>
  <th>Celsius</th>   <td>  647.1550</td> <td>  144.648</td> <td>    4.474</td> <td> 0.000</td> <td>  356.320</td> <td>  937.990</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 9.375</td> <th>  Durbin-Watson:     </th> <td>   1.777</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.009</td> <th>  Jarque-Bera (JB):  </th> <td>   8.756</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.872</td> <th>  Prob(JB):          </th> <td>  0.0126</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.079</td> <th>  Cond. No.          </th> <td>    4.39</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
fig = sm.graphics.plot_partregress_grid(new_test)
fig.tight_layout(pad=1.0)
```


    
![png](output_57_0.png)
    



```python
plot = temp.plot.scatter(x="Celsius", y="NUM_TORNADO", figsize=(15,10))
reg = LinearRegression().fit(temp["Celsius"].values.reshape(-1,1), temp["NUM_TORNADO"])
plt.plot(temp["Celsius"].values.reshape(-1,1), reg.predict(temp["Celsius"].values.reshape(-1,1)))
plot.set_xlabel("Degrees (Celsius)")
plot.set_ylabel("Number of Tornadoes")
plot.set_title("Number of Tornadoes For Each Change in Celsius")
```




    Text(0.5, 1.0, 'Number of Tornadoes For Each Change in Celsius')




    
![png](output_58_1.png)
    


Even with removing the first 20 years of data with the fewest amounts of tornadoes per year, our Prob (F-statistic) number still remains low, at 4.71e-05. Not as small as before, but still shows that there is a dependancy on the number of tornadoes and the global temperature.

#### Global Temperature and Number of Wildfires


```python
df = wildfire_df.groupby("YEAR").size().reset_index(name="counts")
df = pd.merge(df, temp_anamoly_df, on="YEAR")

test = smf.ols(formula="counts ~ Celsius", data=df).fit()
test.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>counts</td>      <th>  R-squared:         </th> <td>   0.011</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>  -0.034</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>  0.2473</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 16 Dec 2020</td> <th>  Prob (F-statistic):</th>  <td> 0.624</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>12:36:35</td>     <th>  Log-Likelihood:    </th> <td> -155.65</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    24</td>      <th>  AIC:               </th> <td>   315.3</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    22</td>      <th>  BIC:               </th> <td>   317.7</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>  230.1528</td> <td>  140.971</td> <td>    1.633</td> <td> 0.117</td> <td>  -62.204</td> <td>  522.509</td>
</tr>
<tr>
  <th>Celsius</th>   <td>  102.7253</td> <td>  206.575</td> <td>    0.497</td> <td> 0.624</td> <td> -325.685</td> <td>  531.136</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>20.213</td> <th>  Durbin-Watson:     </th> <td>   1.497</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  29.917</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.650</td> <th>  Prob(JB):          </th> <td>3.19e-07</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.362</td> <th>  Cond. No.          </th> <td>    8.84</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



#### Global Temperature and Number of Droughts


```python
df = drought_df.groupby("YEAR").size().reset_index(name="counts")
df = pd.merge(df, temp_anamoly_df, on="YEAR")

test = smf.ols(formula="counts ~ Celsius", data=df).fit()
test.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>counts</td>      <th>  R-squared:         </th> <td>   0.001</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>  -0.044</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td> 0.02780</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 16 Dec 2020</td> <th>  Prob (F-statistic):</th>  <td> 0.869</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>12:36:36</td>     <th>  Log-Likelihood:    </th> <td> -212.76</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    24</td>      <th>  AIC:               </th> <td>   429.5</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    22</td>      <th>  BIC:               </th> <td>   431.9</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td> 2563.6192</td> <td> 1522.719</td> <td>    1.684</td> <td> 0.106</td> <td> -594.307</td> <td> 5721.545</td>
</tr>
<tr>
  <th>Celsius</th>   <td> -372.0667</td> <td> 2231.345</td> <td>   -0.167</td> <td> 0.869</td> <td>-4999.592</td> <td> 4255.459</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>17.389</td> <th>  Durbin-Watson:     </th> <td>   0.730</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  19.022</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.700</td> <th>  Prob(JB):          </th> <td>7.40e-05</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.733</td> <th>  Cond. No.          </th> <td>    8.84</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



#### Global Temperature and Number of Tropical Cyclones


```python
# Atlantic Tropical Cyclones
atlantic_df = atlantic_cyclone_df.groupby("YEAR").size().reset_index(name="counts")
atlantic_df = pd.merge(atlantic_df, temp_anamoly_df, on="YEAR")

test = smf.ols(formula="counts ~ Celsius", data=atlantic_df).fit()
test.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>counts</td>      <th>  R-squared:         </th> <td>   0.251</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.246</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   46.33</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 16 Dec 2020</td> <th>  Prob (F-statistic):</th> <td>2.81e-10</td>
</tr>
<tr>
  <th>Time:</th>                 <td>12:55:48</td>     <th>  Log-Likelihood:    </th> <td> -887.66</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   140</td>      <th>  AIC:               </th> <td>   1779.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   138</td>      <th>  BIC:               </th> <td>   1785.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>  328.3399</td> <td>   11.886</td> <td>   27.623</td> <td> 0.000</td> <td>  304.837</td> <td>  351.843</td>
</tr>
<tr>
  <th>Celsius</th>   <td>  229.5208</td> <td>   33.721</td> <td>    6.806</td> <td> 0.000</td> <td>  162.843</td> <td>  296.198</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 7.175</td> <th>  Durbin-Watson:     </th> <td>   1.580</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.028</td> <th>  Jarque-Bera (JB):  </th> <td>   6.848</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.522</td> <th>  Prob(JB):          </th> <td>  0.0326</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.290</td> <th>  Cond. No.          </th> <td>    2.90</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
year = pd.cut(atlantic_df["YEAR"], 7)
atlantic_df["year_binned"] = year

atlantic_df["residual"] = test.resid

atlantic_df
plt.figure(figsize=(15,10))
sns.violinplot(x=atlantic_df["year_binned"], y=atlantic_df["residual"])
plt.title("Residuals over Time for Atlantic Tropical Cyclones")
plt.show()
plt.close()
```


    
![png](output_66_0.png)
    



```python
# Pacific Tropical Cyclones
pacific_df = pacific_cyclone_df.groupby("YEAR").size().reset_index(name="counts")
pacific_df = pd.merge(pacific_df, temp_anamoly_df, on="YEAR")

test = smf.ols(formula="counts ~ Celsius", data=df).fit()
test.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>counts</td>      <th>  R-squared:         </th> <td>   0.336</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.326</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   34.93</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 16 Dec 2020</td> <th>  Prob (F-statistic):</th> <td>1.17e-07</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:03:47</td>     <th>  Log-Likelihood:    </th> <td> -470.46</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    71</td>      <th>  AIC:               </th> <td>   944.9</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    69</td>      <th>  BIC:               </th> <td>   949.4</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>  279.5575</td> <td>   31.045</td> <td>    9.005</td> <td> 0.000</td> <td>  217.624</td> <td>  341.491</td>
</tr>
<tr>
  <th>Celsius</th>   <td>  424.5233</td> <td>   71.828</td> <td>    5.910</td> <td> 0.000</td> <td>  281.231</td> <td>  567.815</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 4.716</td> <th>  Durbin-Watson:     </th> <td>   1.359</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.095</td> <th>  Jarque-Bera (JB):  </th> <td>   4.648</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.587</td> <th>  Prob(JB):          </th> <td>  0.0979</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.559</td> <th>  Cond. No.          </th> <td>    3.60</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
year = pd.cut(pacific_df["YEAR"], 7)
pacific_df["year_binned"] = year

pacific_df["residual"] = test.resid

plt.figure(figsize=(15,10))
sns.violinplot(x=pacific_df["year_binned"], y=pacific_df["residual"])
plt.title("Residuals over Time for Pacific Tropical Cyclones")
plt.show()
plt.close()
```


    
![png](output_68_0.png)
    


<hr>
<h2><center>Insight and Recommendations</center></h2>

Through the use of the Data Science pipeline, we have been able to observe trends relating to the rising global temperature and the effect that it brings on natural disasters. From the data that we have observed and the mathematical models that we created during our analysis phase, we have come to the following conclusions:

##### Tornadoes in the United States
As stated throughout this project, there is some bias associated with the data that we collected from NOAA. Specifically, some tornadoes may have not been represented in the dataframe, or may have even been counted multiple times from different observers monitoring the storm. However, after correcting for these errors, we still came to the conclusion that the number of tornadoes has risen over the years as the Earth begins to heat up. Based on our model, this will continue to do so if the planet continues to warm.

##### Tropical Cyclones 
We also see that there has been a rise in tropical cyclones which can be more accurately reported, since these events are often massive and can be seen from space and with weather radar. From the mathematical model that we created, the number of tropical cyclones in the Atalantic and Pacific Ocean has continued to rise, and will do so as the Earth continues to warm.

##### Droughts and Wildfires in the United States
According to the examined data and models, it does not appear that these two natural disasters are necessarily tied to global warming due to their high P-statistic. The two are closely related to each other based on their graphs, and thus should be placed on the back burner for now. Potential bias still exists, however, with the lack of data that was presented. The recorded data only dates back to 1996, which may or may not be enough time to witness these trends. They may also both be related to global warming, but will possibly increase with further warming of the planet.

##### Final Recommendations
Further monitoring of the Earth's temperature and natural disasters is required. Although some of the data remains inconclusive, they may change if the temperature continues to steadily rise.

