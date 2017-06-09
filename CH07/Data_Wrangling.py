import pandas as pd
from pandas import DataFrame
from pandas import Series

import numpy as np


""" Database-style DataFrame Merges """
# Merge or join operations combine data sets by linking rows using one or more keys.
# These operations are central to relational databases.
df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 
				 'data1': range(7)})

df2 = DataFrame({'key': ['a', 'b', 'd'],
				 'data2': range(3)})

# This is an example of a many-to-one merge situation
pd.merge(df1, df2)

# specify which column to join on explicitly
pd.merge(df1, df2, on='key')

# If the column names are different in each object, you can specify them separately:
df3 = DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
				 'data1': range(7)})

df4 = DataFrame({'rkey': ['a', 'b', 'd'],
				 'data2': range(3)})

pd.merge(df3, df4, left_on='lkey', right_on='rkey')

# Other possible options are  'left' ,  'right' , and  'outer' .
pd.merge(df1, df2, how='outer')
pd.merge(df1, df2, how='left')
pd.merge(df1, df2, how='right')

# Many-to-many merges
df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
				 'data1': range(6)})

df2 = DataFrame({'key': ['a', 'b', 'a', 'b', 'd'],
				 'data2': range(5)})

pd.merge(df1, df2, on='key', how='left')

pd.merge(df1, df2, how='inner')

# To merge with multiple keys, pass a list of column names:
left = DataFrame({'key1': ['foo', 'foo', 'bar'],
				  'key2': ['one', 'two', 'one'],
				  'lval': [1, 2, 3]})

right = DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
				   'key2': ['one', 'one', 'one', 'two'],
				   'rval': [4, 5, 6, 7]})

pd.merge(left, right, on=['key1', 'key2'], how='outer')

# the treatment of overlapping column names:
# merge has a  suffixes option for specifying strings to append to overlapping
# names in the left and right DataFrame objects:
pd.merge(left, right, on='key1')
pd.merge(left, right, on='key1', suffixes=('_left', '_right'))


""" Merging on Index 
---------------------
In some cases, the merge key or keys in a DataFrame will be found in its index. In this
case, you can pass  left_index=True or  right_index=True (or both) to indicate that the
index should be used as the merge key """
left1 = DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'],
				   'value': range(6)})

right1 = DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])

# Use 'left_on' instead of 'on' to specify the column name
pd.merge(left1, right1, left_on='key', right_index=True)
pd.merge(left1, right1, left_on='key', right_index=True, how='outer')

# By default, DataFrame’s  join method performs a left join on the join keys
left1.join(right1, on='key') 

# With hierarchically-indexed data, things are a bit more complicated:
lefth = DataFrame({'key1': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
				   'key2': [2000, 2001, 2002, 2001, 2002],
				   'data': np.arange(5.)})

righth = DataFrame(np.arange(12).reshape((6, 2)),
				   index=[['Nevada', 'Nevada', 'Ohio', 'Ohio', 'Ohio', 'Ohio'],
						  [2001, 2000, 2000, 2000, 2001, 2002]],
				   columns=['event1', 'event2'])

pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True)

pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True, how='outer')

# Using the indexes of both sides of the merge is also not an issue:
left2 = DataFrame([[1., 2.], [3., 4.], [5., 6.]], index=['a', 'c', 'e'],
				  columns=['Ohio', 'Nevada'])

right2 = DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],
				  index=['b', 'c', 'd', 'e'], columns=['Missouri', 'Alabama'])

pd.merge(left2, right2, how='outer', left_index=True, right_index=True)

left2.join(right2, how='outer')

# Pass a list of DataFrames to join
another = DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]],
				    index=['a', 'c', 'e', 'f'], columns=['New York', 'Oregon'])

left2.join([right2, another])
left2.join([right2, another], how='outer')



""" Concatenating Along an Axis """
# NumPy has a  concatenate function for doing this with raw NumPy arrays:
arr = np.arange(12).reshape((3, 4))
np.concatenate([arr, arr], axis=1)

"""In the context of pandas objects such as Series and DataFrame, having labeled axes
enable you to further generalize array concatenation. In particular, you have a number
of additional things to think about:
• If the objects are indexed differently on the other axes, should the collection of
axes be unioned or intersected?
• Do the groups need to be identifiable in the resulting object?
• Does the concatenation axis matter at all?
The 'concat' function in pandas provides a consistent way to address each of these con-
cerns. """

s1 = Series([0, 1], index=['a', 'b'])
s2 = Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = Series([5, 6], index=['f', 'g'])

# Calling concat with these object in a list glues together the values and indexes:
pd.concat([s1, s2, s3])

# By default  concat works along  axis=0 , producing another Series. If you pass  axis=1 , 
# the result will instead be a DataFrame ( axis=1 is the columns):
pd.concat([s1, s2, s3], axis=1)

# In this case there is no overlap on the other axis, which as you can see is the sorted
# union (the  'outer' join) of the indexes. You can instead intersect them by passing join='inner' 
s4 = pd.concat([s1 * 5, s3])
pd.concat([s1, s4], axis=1, join='inner')

# You can even specify the axes to be used on the other axes with join_axes :
pd.concat([s1, s4], axis=1, join_axes=[['a', 'c', 'b', 'e']])

# Create a hierarchical index on the concatenation axis, use the keys argument:
result = pd.concat([s1, s1, s3], keys=['one', 'two', 'three'])

# In the case of combining Series along axis=1, the keys become the DataFrame column headers:
pd.concat([s1, s2, s3], axis=1, keys=['one', 'two', 'three'])

# The same logic extends to DataFrame objects:
df1 = DataFrame(np.arange(6).reshape(3, 2), index=['a', 'b', 'c'],
			    columns=['one', 'two'])

df2 = DataFrame(5 + np.arange(4).reshape(2, 2), index=['a', 'c'],
				columns=['three', 'four'])

pd.concat([df1, df2], axis=1, keys=['level1', 'level2'])

# If you pass a dict of objects instead of a list, the dict’s keys will be used for the keys option:
pd.concat({'level1': df1, 'level2': df2}, axis=1)

# Additional arguments governing
pd.concat([df1, df2], axis=1, keys=['level1', 'level2'],
		  names=['upper', 'lower'])

# In which the row index is not meaningful in the context of the analysis,
# you can pass ignore_index=True :
df1 = DataFrame(np.random.randn(3, 4), columns=['a', 'b', 'c', 'd'])
df2 = DataFrame(np.random.randn(2, 3), columns=['b', 'd', 'a'])
pd.concat([df1, df2], ignore_index=True)


""" Combining Data with Overlap """
# Consider NumPy’s  where function, which expressed a vectorized if-else:
a = Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan],
			index=['f', 'e', 'd', 'c', 'b', 'a'])

b = Series(np.arange(len(a), dtype=np.float64),
		   index=['f', 'e', 'd', 'c', 'b', 'a'])

b[-1] = np.nan

# Series has a combine_first method, which performs the equivalent of this operation plus data alignment:
b[:-2].combine_first(a[2:])

# With DataFrames, combine_first naturally does the same thing column by column:
df1 = DataFrame({'a': [1., np.nan, 5., np.nan],
				 'b': [np.nan, 2., np.nan, 6.],
				 'c': range(2, 18, 4)})

df2 = DataFrame({'a': [5., 4., np.nan, 3., 7.],
				 'b': [np.nan, 3., 4., 6., 8.]})

df1.combine_first(df2)


""" Reshaping """
data = DataFrame(np.arange(6).reshape((2, 3)),
				 index=pd.Index(['Ohio', 'Colorado'], name='state'),
				 columns=pd.Index(['one', 'two', 'three'], name='number'))

# Using the stack method on this data pivots the columns into the rows, producing a Series:
result = data.stack()
result

# From a hierarchically-indexed Series, you can rearrange the data back into a DataFrame with unstack :
result.unstack()

# You can unstack a different level by passing a level number or name:
result.unstack(0)
result.unstack('state')

# Unstacking might introduce missing data if all of the values in the level aren’t found in
# each of the subgroups:
s1 = Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
s2 = Series([4, 5, 6], index=['c', 'd', 'e'])

data2 = pd.concat([s1, s2], keys=['one', 'two'])
data2.unstack()

# Stacking filters out missing data by default, so the operation is easily invertible:
data2.unstack().stack()
data2.unstack().stack(dropna=False)

# When unstacking in a DataFrame, the level unstacked becomes the lowest level in the result:
df = DataFrame({'left': result, 'right': result + 5},
			   columns=pd.Index(['left', 'right'], name='side'))

df.unstack('state') 
df.unstack('state').stack('side')


""" Data Transformation """
# The DataFrame method duplicated returns a boolean Series indicating whether each
# row is a duplicate or not:
data = DataFrame({'k1': ['one'] * 3 + ['two'] * 4,
				  'k2': [1, 1, 2, 3, 3, 4, 4]})

data.duplicated()

# drop_duplicates returns a DataFrame where the duplicated array is False:
data.drop_duplicates()

#  filter duplicates only based on the  'k1' column:
data['v1'] = range(7)
data.drop_duplicates(['k1'])

# duplicated and  drop_duplicates by default keep the first observed value combination.
# Passing keep='last' will return the last one:
data.drop_duplicates(['k1', 'k2'], keep='last')


# Transforming Data Using a Function or Mapping
# The map method on a Series accepts a function or dict-like object containing a mapping
data = DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami',
				 'corned beef', 'Bacon', 'pastrami', 'honey ham',
				 'nova lox'],
				 'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})

meat_to_animal = {
	'bacon': 'pig',
	'pulled pork': 'pig',
	'pastrami': 'cow',
	'corned beef': 'cow',
	'honey ham': 'pig',
	'nova lox': 'salmon'
}

# my opinion: map() actually is a method of substitution
data['animal'] = data['food'].map(str.lower).map(meat_to_animal)

# Replacing Values
data = Series([1., -999., 2., -999., -1000., 3.])

data.replace(-999, np.nan)
data.replace([-999, -1000], np.nan)
data.replace([-999, -1000], [np.nan, 0])
data.replace({-999: np.nan, -1000: 0})


""" Renaming Axis Indexes """
data = DataFrame(np.arange(12).reshape((3, 4)),
				 index=['Ohio', 'Colorado', 'New York'],
				 columns=['one', 'two', 'three', 'four'])

# Like a Series, the axis indexes have a map method:
data.index.map(str.upper)

# You can assign to index, modifying the DataFrame in place:
data.index = data.index.map(str.upper)

# If you want to create a transformed version of a data set without modifying the original,
# a useful method is rename 
data.rename(index=str.title, columns=str.upper)

# rename can be used in conjunction with a dict-like object providing new values
# for a subset of the axis labels:
data.rename(index={'OHIO': 'INDIANA'},
			columns={'three': 'peekaboo'})

#  Should you wish to modify a data set in place, pass inplace=True :
data.rename(index={'OHIO': 'INDIANA'}, inplace=True)


""" Discretization and Binning """
# group into discrete age buckets:
# use 'cut' function
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]

cats = pd.cut(ages, bins)
cats.labels

pd.value_counts(cats)

# With mathematical notation for intervals
# Which side is closed can be changed by passing 'right=False'
pd.cut(ages, [18, 26, 36, 61, 100], right=False)

# You can also pass your own bin names by passing a list or array to the labels option:
group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
pd.cut(ages, bins, labels=group_names)

# If you pass  cut a integer number of bins instead of explicit bin edges, 
# it will compute equal-length bins based on the minimum and maximum values in the data.
data = np.random.rand(20)
pd.cut(data, 4, precision=2)

# using cut will not usually result in each bin having the same number of data points:
# Since qcut uses sample quantiles instead, by definition you will obtain roughly equal-size bins:
data = np.random.randn(1000)
cats = pd.qcut(data, 4)
pd.value_counts(cats)

# Similar to  cut you can pass your own quantiles
cats = pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.])
pd.value_counts(cats)


""" Detecting and Filtering Outliers """
np.random.seed(12345)
data = DataFrame(np.random.randn(1000, 4))

# find values in one of the columns exceeding three in magnitude:
col = data[3]
col[np.abs(col) > 3]

# To select all rows having a value exceeding 3 or -3, you can use the any method on a
# boolean DataFrame:
data[(np.abs(data) > 3).any(1)]

# cap values outside the interval -3 to 3:
# np.sign returns an array of 1 and -1 depending on the sign of the values
data[np.abs(data) > 3] = np.sign(data) * 3
data.describe()


""" Permutation and Random Sampling """
df = DataFrame(np.arange(5 * 4).reshape(5, 4))

# Calling  permutation with the length of the axis you want to permute 
# produces an array of integers indicating the new ordering:
sampler = np.random.permutation(5)

# That array can then be used in ix-based indexing or the  take function:
df.take(sampler)

# To select a random subset without replacement
df.take(np.random.permutation(len(df))[:3])

# To generate a sample with replacement
bag = np.array([5, 7, -1, 6, 4])
sampler = np.random.randint(0, len(bag), size=10)
draws = bag.take(sampler)
draws

""" Computing Indicator/Dummy Variables """
df = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
				'data1': range(6)})

# Using the get_dummies will create a new column for every unique string in a certain column
pd.get_dummies(df['key'])

#  add a prefix to the columns in the indicator DataFrame
dummies = pd.get_dummies(df['key'], prefix='key')
df_with_dummy = df[['data1']].join(dummies)


# If a row in a DataFrame belongs to multiple categories, 
# things are a bit more complicated
mnames = ['movie_id', 'title', 'genres']

movies = pd.read_table('ch07/movies.dat', sep='::', header=None,
						names=mnames)

genre_iter = (set(x.split('|')) for x in movies.genres)
genres = sorted(set.union(*genre_iter))

dummies = DataFrame(np.zeros((len(movies), len(genres))), columns=genres)

for i, gen in enumerate(movies.genres):
	dummies.ix[i, gen.split('|')] = 1

movies_windic = movies.join(dummies.add_prefix('Genre_'))
movies_windic.ix[0]

# A useful recipe for statistical applications is to combine get_dummies 
# with a discretization function like cut :
values = np.random.rand(10)
bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
pd.get_dummies(pd.cut(values, bins))


""" String Manipulation """
# a comma-separated string can be broken into pieces with 'split':
val = 'a,b, guido'
val.split(',')

# split is often combined with  strip to trim whitespace (including newlines):
pieces = [x.strip() for x in val.split(',')]

# These substrings could be concatenated together with a two-colon delimiter using addition:
first, second, third = pieces
first + '::' + second + '::' + third

# A faster and more Pythonic way is to pass a list or tuple to the  join method on the string  '::'
'::'.join(pieces)

# Other methods are concerned with locating substrings.
'guido' in val
val.index(',') 
val.find(':')

#  count returns the number of occurrences of a particular substring:
val.count(',')

# replace will substitute occurrences of one pattern for another
val.replace(',', '::') 
val.replace(',', '')


""" Regular expressions """
import re

text = "foo bar\t baz \tqux"
re.split('\s+', text)

# You can compile the regex yourself with re.compile ,forming a reusable regex object:
regex = re.compile('\s+')
regex.split(text)

# get a list of all patterns matching the regex, you can use the findall method:
regex.findall(text)

# findall returns all matches in a string,  
# search returns only the first match
text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""

pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
regex = re.compile(pattern, flags=re.IGNORECASE)

regex.findall(text)


# search returns a special match object for the first email address in the text. For the
# above regex, the match object can only tell us the start and end position of the pattern
# in the string
m = regex.search(text)
m
text[m.start():m.end()]

# regex.match returns  None, as it only will match if the pattern occurs at the start of the string:
print(regex.match(text))

# sub will return a new string with occurrences of the pattern replaced by the a new string:
print regex.sub('REDACTED', text)

# find email addresses and simultaneously segment each address into its 3 components:
pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
regex = re.compile(pattern, flags=re.IGNORECASE)
m = regex.match('wesm@bright.net')
m.groups()

# findall returns a list of tuples when the pattern has groups:
regex.findall(text)

# sub also has access to groups in each match using special symbols like  \1, \2 , etc.
print(regex.sub(r'Username: \1, Domain: \2, Suffix: \3', text))

# one variation on the above email regex gives names to the match groups:
regex = re.compile(r"""
(?P<username>[A-Z0-9._%+-]+)
@
(?P<domain>[A-Z0-9.-]+)
\.
(?P<suffix>[A-Z]{2,4})""", flags=re.IGNORECASE|re.VERBOSE)

m = regex.match('wesm@bright.net')
m.groupdict()


""" Vectorized string functions in pandas """
data = {'Dave': 'dave@google.com', 'Steve': 'steve@gmail.com',
		'Rob': 'rob@gmail.com', 'Wes': np.nan}

data
data.isnull()

# check whether each email address has 'gmail' in it with str.contains :
data.str.contains('gmail')

# Regular expressions can be used, too, along with any  re options like IGNORECASE :
pattern
data.str.findall(pattern, flags=re.IGNORECASE)

# There are a couple of ways to do vectorized element retrieval. Either use  str.get or index into the  str attribute:
matches = data.str.match(pattern, flags=re.IGNORECASE)
matches.str.get(1)
matches.str[0]
data.str[:5]


""" Example: USDA Food Database """
import json

db = json.load(open('ch07/foods-2011-10-03.json'))

# Each entry in  db is a dict containing all the data for a single food:
db[0].keys() 

# The 'nutrients' field is a list of dicts, one for each nutrient:
db[0]['nutrients'][0]

nutrients = DataFrame(db[0]['nutrients'])

# When converting a list of dicts to a DataFrame, we can specify a list of fields to extract:
info_keys = ['description', 'group', 'id', 'manufacturer']
info = DataFrame(db, columns=info_keys)
info[:5]

# You can see the distribution of food groups with  value_counts 
pd.value_counts(info.group)[:10]

# to assemble the nutrients for each food into a single large table
nutrients = []

for rec in db:
	fnuts = DataFrame(rec['nutrients'])
	fnuts['id'] = rec['id']
	nutrients.append(fnuts)

nutrients = pd.concat(nutrients, ignore_index=True)

# there are duplicates in this DataFrame, so it makes
# things easier to drop them:
nutrients.duplicated().sum()
nutrients = nutrients.drop_duplicates()

# Since 'group' and  'description' is in both DataFrame objects (with different contents), 
# we can rename them to make it clear what is what:
nutrients['group'].drop_duplicates()
info['group'].drop_duplicates()
nutrients['description'].drop_duplicates()
info['description'].drop_duplicates()

col_mapping = {'description' : 'nutrient', 'group' : 'nutgroup'}
nutrients = nutrients.rename(columns=col_mapping, copy=False)
nutrients.columns

# we’re ready to merge 'info' with 'nutrients' :
ndata = pd.merge(nutrients, info, on='id', how='outer')
ndata.ix[30000]