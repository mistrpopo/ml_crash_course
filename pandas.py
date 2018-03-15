import pandas as pd
import numpy as np

#create series and data frames
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
cities = pd.DataFrame({ 'City name': city_names, 'Population': population })

#load data frames from csv
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

#print first 5 records of data frame
california_housing_dataframe.head()
#print histogram (requires numpy/matplotlib/stuff)
california_housing_dataframe.hist('housing_median_age') #can fail depending on pandas version
#compute and show simple statistics on each series
california_housing_dataframe.describe()

#compatibility of pandas Series with numpy
np.log(population) #calculate natural log (ln) of each population

#Series.apply accepts as an argument a lambda
population.apply(lambda val: val > 1000000)

#extend an existing DataFrame with an additional series
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
#simple operation on series
cities['Population density'] = cities['Population'] / cities['Area square miles']

#exercise 1 : Modify the cities table by adding a new boolean column that is True if and only if both of the following are True:

    The city is named after a saint.
    The city has an area greater than 50 square miles.

city_name_saint = city_names.str.split(' ').str[0] == "San"
city_area_50sqm = cities['Area square miles'] > 50
cities['ex1'] = city_name_saint & city_area_50sqm