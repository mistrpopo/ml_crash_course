import numpy as np
import pandas as pd

#create series and data frames
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

# the Series is the data structure for a single column of a DataFrame, not only conceptually, but literally, 
# i.e. the data in a DataFrame is actually stored in memory as a collection of Series. 
cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
#   City name    	Population
# 0 San Francisco  	852469
# 1 San Jose 	    1015785
# 2 Sacramento 	    485199

print(type(cities['City name']))
# <class 'pandas.core.series.Series'>

print(cities['City name'])
# 0    San Francisco
# 1         San Jose
# 2       Sacramento
# Name: City name, dtype: object

print(type(cities['City name'][1]))
# <class 'str'>

print(cities['City name'][1])
# 'San Jose'

print(type(cities[0:2]))
# <class 'pandas.core.frame.DataFrame'>
print(cities[0:2])
# 		City name 		Population
# 0 	San Francisco 	852469
# 1 	San Jose 		1015785

#compatibility of pandas Series with numpy
print(np.log(population)) #calculate natural log (ln) of each population

#Series.apply accepts as an argument a lambda and returns a new panda Series
print(population.apply(lambda val: val > 1000000))
print(type(population.apply(lambda val: val > 1000000)))

#extend an existing DataFrame with an additional series
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
#simple operation on series
cities['Population density'] = cities['Population'] / cities['Area square miles']

#load data frames from csv
if False:
    california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
    print(type(california_housing_dataframe))
    # <class 'pandas.core.frame.DataFrame'>

    #print first 5 records of data frame
    california_housing_dataframe.head()
    #   	longitude 	latitude 	housing_median_age 	total_rooms 	total_bedrooms 	population 	households 	median_income 	median_house_value
    # 0 	-114.31 	34.19 		15.0			 	5612.0		 	1283.0		 	1015.0	 	472.0	 	1.4936 			66900.0
    # 1 	-114.47 	34.40 		19.0			 	7650.0		 	1901.0		 	1129.0	 	463.0	 	1.8200 			80100.0
    # 2 	-114.56 	33.69 		17.0			 	720.0		 	174.0		 	333.0	 	117.0	 	1.6509 			85700.0
    # 3 	-114.57 	33.64 		14.0			 	1501.0		 	337.0		 	515.0	 	226.0	 	3.1917 			73400.0
    # 4 	-114.57 	33.57 		20.0			 	1454.0		 	326.0		 	624.0	 	262.0	 	1.9250 			65500.0

    #print histogram (requires numpy/matplotlib/stuff)
    california_housing_dataframe.hist('housing_median_age')
    print(type(california_housing_dataframe.hist('housing_median_age'))) #can fail depending on pandas version
    # <class 'numpy.ndarray'>

    #compute and show simple statistics on each series
    california_housing_dataframe.describe()
    #           longitude       latitude 	    housing_median_age 	total_rooms 	total_bedrooms 	population  	households  	median_income 	median_house_value
    # count 	17000.000000 	17000.000000 	17000.000000 	    17000.000000 	17000.000000 	17000.000000 	17000.000000 	17000.000000 	17000.000000
    # mean 	    -119.562108 	35.625225   	28.589353       	2643.664412 	539.410824  	1429.573941 	501.221941  	3.883578 	    207300.912353
    # std 	    2.005166 	    2.137340 	    12.586937       	2179.947071 	421.499452  	1147.852959 	384.520841  	1.908157    	115983.764387
    # min 	    -124.350000 	32.540000   	1.000000 	        2.000000 	    1.000000 	    3.000000 	    1.000000 	    0.499900    	14999.000000
    # 25%   	-121.790000 	33.930000   	18.000000       	1462.000000 	297.000000  	790.000000  	282.000000  	2.566375 	    119400.000000
    # 50% 	    -118.490000 	34.250000   	29.000000       	2127.000000 	434.000000  	1167.000000 	409.000000  	3.544600 	    180400.000000
    # 75% 	    -118.000000 	37.720000   	37.000000       	3151.250000 	648.250000 	    1721.000000 	605.250000  	4.767000 	    265000.000000
    # max 	    -114.310000 	41.950000   	52.000000       	37937.000000 	6445.000000 	35682.000000 	6082.000000 	15.000100   	500001.000000
    print(type(california_housing_dataframe.describe()))
    # <class 'pandas.core.frame.DataFrame'>


#exercise 1 : Modify the cities table by adding a new boolean column that is True if and only if both of the following are True:
#    The city is named after a saint.
#    The city has an area greater than 50 square miles.

city_name_saint = city_names.str.split(' ').str[0] == "San"
city_area_50sqm = cities['Area square miles'] > 50
cities['ex1'] = city_name_saint & city_area_50sqm
print(cities['ex1'])
# 0    False
# 1     True
# 2    False
# Name: ex1, dtype: bool



# the index property assigns an identifier value to each Series item or DataFrame row. 
print(city_names.index)

# Call 'reindex' to manually reorder the rows
city_names.reindex([2, 0, 1])
#Reindexing is a great way to shuffle (randomize) a DataFrame.
cities.reindex(np.random.permutation(cities.index))

# If your reindex input array includes values not in the original DataFrame index values, reindex will add new rows for these "missing" indices 
# and populate all corresponding columns with NaN values
print(cities.reindex([0, 4, 5, 2]))
#        City name  Population  Area square miles  Population density    ex1
# 0  San Francisco    852469.0              46.87        18187.945381  False
# 4            NaN         NaN                NaN                 NaN    NaN
# 5            NaN         NaN                NaN                 NaN    NaN
# 2     Sacramento    485199.0              97.92         4955.055147  False


