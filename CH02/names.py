import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

names1880 = pd.read_csv('C:/Users/Administrator/Desktop/yob1880.txt', 
							names=['name', 'sex', 'births'])

# Use the sum of the births column by sex as the total number of births in that year:
names1880.groupby('sex').births.sum()

# Assemble all of the data into a single DataFrame and further to add a year field:
years = range(1880, 2011)
pieces = []
columns = ['name', 'sex', 'births']

for year in years:
	path = 'C:/Users/Administrator/Desktop/ch02/names/yob%d.txt' %year
	frame = pd.read_csv(path, names=columns)
	frame['year'] = year
	pieces.append(frame)

# Concatenate everything into a single DataFrame:
names = pd.concat(pieces, ignore_index=True)
""" ignore_index=True: clear the original index and reset it
	ignore_index=False: preserve the original index"""

# Aggregating the data at the year and sex level using groupby or pivot_table:
total_births = names.pivot_table('births', index='year', columns='sex', aggfunc=sum) 
total_births.tail()

total_births.plot(title='Total births by sex and year')

def add_prop(group):
	# Integer division floors
	births = group.births.astype(float)
	group['prop'] = births / births.sum()
	return group

names = names.groupby(['year', 'sex']).apply(add_prop)

# Verifying that the  prop column sums to 1 within all the groups:
np.allclose(names.groupby(['year', 'sex']).prop.sum(), 1)

# Extract the top 1000 names for each sex/year combination:
def get_top1000(group):
	return group.sort_index(by='births', ascending=False)[:1000]

top1000 = names.groupby(['year', 'sex']).apply(get_top1000)

# Splitting the Top 1,000 names into the boy and girl portions:
boys = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']

total_births = top1000.pivot_table('births', index='year', columns='name',
									aggfunc=sum)

subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]

subset.plot(subplots=True, figsize=(12, 10), grid=False,
			title="Number of births per year")

plt.show()


"""
One explanation for the decrease in plots above is that fewer parents are choosing
common names for their children. This hypothesis can be explored and confirmed in
the data. One measure is the proportion of births represented by the top 1000 most
popular names, which I aggregate and plot by year and sex:
"""
table = top1000.pivot_table('prop', index='year', columns='sex', aggfunc=sum)

table.plot(title="Sum of table1000.prop by year and sex",
			yticks=np.linspace(0, 1.2, 13), xticks=range(1880, 2020, 10))
# np.linspace: return evenly spaced numbers over a specified intervals.
plt.show()


"""
Another interesting met-
ric is the number of distinct names, taken in order of popularity from highest to lowest,
in the top 50% of births. 
"""
df = boys[boys.year == 2010]

"""
After sorting  prop in descending order, find out how many of the most popular
names it takes to reach 50%
"""
prop_cumsum = df.sort_index(by='prop', ascending=False).prop.cumsum()
prop_cumsum.searchsorted(0.5)

# By contrast, in 1900 this number was much smaller:
df = boys[boys.year == 1900]
in1990 = df.sort_index(by='prop', ascending=False).prop.cumsum()
# Since arrays are zero-indexed, adding 1 to this result
in1990.searchsorted(0.5) + 1

"""
It should now be fairly straightforward to apply this operation to each year/sex com-
bination;  groupby those fields and  apply a function returning the count for each group:
"""
def get_quantile_count(group, q=0.5):
	group = group.sort_index(by='prop', ascending=False)
	return int(group.prop.cumsum().searchsorted(q) + 1)

diversity = top1000.groupby(['year', 'sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')

diversity.plot(title="Number of popular names in top 50%")


"""
In 2007, a baby name researcher Laura Wattenberg pointed out on her website that 
the distribution of boy names by final letter has changed significantly over the last 100 years. 
To see this, I first aggregate all of the births in the full data set by year, sex, and final letter:
"""
# extract last letter from name column
get_last_letter = lambda x: x[-1]
last_letters = names.name.map(get_last_letter)
last_letters.name = 'last_letter'

table = names.pivot_table('births', index=last_letters, columns=['sex', 'year'], aggfunc=sum)

# Then, select out three representative years spanning the history and print the first few rows:
subtable = table.reindex(columns=[1910, 1960, 2010], level='year')

subtable.sum()
letter_prop = subtable / subtable.sum().astype(float)

# Make bar plots for each sex broken down by year:
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
letter_prop['M'].plot(kind='bar', rot=0, ax=axes[0], title='Male')
letter_prop['F'].plot(kind='bar', rot=0, ax=axes[1], title='Female', legend=False)

"""
Going back to the full table created above, I again normalize by year and sex
and select a subset of letters for the boy names, finally transposing to make each column
a time series:
"""
letter_prop = table / table.sum().astype(float)
dny_ts = letter_prop.ix[['d', 'n', 'y'], 'M'].T
dny_ts.head()
dny_ts.plot()
plt.show()


"""Boy names that became girl names (and vice versa)"""
all_names = top1000.name.unique()
mask = np.array(['lesl' in x.lower() for x in all_names])
lesley_like = all_names[mask]

# Filter down to just those names and sum births grouped by name to see the relative frequencies:
filtered = top1000[top1000.name.isin(lesley_like)]
filtered.groupby('name').births.sum()

# Aggregate by sex and year and normalize within year:
table = filtered.pivot_table('births', index='year', columns='sex', aggfunc='sum')
table = table.div(table.sum(1), axis=0)
table.tail()
table.plot(style={'M': 'k-', 'F': 'k--'})
plt.show()