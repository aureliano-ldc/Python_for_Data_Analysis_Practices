import json

from pandas import DataFrame, Series
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

path = 'C:/Users/Administrator/Desktop/usagov_bitly_data2012-03-16-1331923249.txt'
records = [json.loads(line) for line in open(path)]

frame = DataFrame(records)

# The fillna function can replace missing (NA) values and unknown(empty strings) values can be replaced by boolean array indexing:
clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'

# The Series object returned by frame['tz'] has a method value_counts that gives us frequencies of time_zones:
tz_counts = clean_tz.value_counts()

# Look at the top 10 common time_zones
tz_counts[:10]

# Making a horizontal bar plot can be accomplished using the plot method on the tz_counts objects:
tz_counts[:10].plot(kind='barh', rot=0)

# Show the plot (a specific step for windows?)
plt.show()

# The a field contains information about the browser, device, or application used to perform the URL shortening:
frame['a'][1]
frame['a'][50]
frame['a'][51]

# Split off the first token in the string
results = Series([x.split()[0] for x in frame.a.dropna()])
results[:5]
results.value_counts()[:8]

# Exclude missing agents from the data
cframe = frame[frame.a.notnull()] 

# Compute a value whether each row is Windows or not:
operating_system = np.where(cframe['a'].str.contains('Windows'),
							'Windows', 'Not Windows')

operating_system[:5]

"""
About groupby() and it's size() method, visit http://www.cnblogs.com/splended/p/5278078.html
for explanation;
About stack() and unstack(), visit http://pandas.pydata.org/pandas-docs/stable/10min.html;
"""
by_tz_os = cframe.groupby(['tz', operating_system])
agg_counts = by_tz_os.size().unstack().fillna(0)
agg_counts[:10]

# Use to sort in ascending order
indexer = agg_counts.sum(1).argsort()
indexer[:10]

# use take to select the rows in that order, then slice off the last 10 rows:
count_subset = agg_counts.take(indexer)[-10:]

# Plot
count_subset.plot(kind='barh', stacked=True)
"""
The plot doesn’t make it easy to see the relative percentage of Windows users in the
smaller groups, but the rows can easily be normalized to sum to 1 then plotted again
"""
normed_subset = count_subset.div(count_subset.sum(1), axis=0)
normed_subset.plot(kind='barh', stacked=True)