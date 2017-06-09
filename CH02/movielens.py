"""
The MovieLens 1M data set contains 1 million ratings collected from 6000 users on
4000 movies. It’s spread across 3 tables: ratings, user information, and movie infor-
mation. After extracting the data from the zip file, each table can be loaded into a pandas
DataFrame object using  pandas.read_table :
"""

import pandas as pd

unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table('C:/Users/Administrator/Desktop/users.dat', 
						sep='::', header=None, names=unames)

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('C:/Users/Administrator/Desktop/ratings.dat',
							sep='::', header=None,names=rnames)

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('C:/Users/Administrator/Desktop/movies.dat',
							sep='::', header=None, names=mnames)

"""
Using pandas’s merge function, we first merge ratings with
users then merging that result with the movies data. pandas infers which columns to
use as the merge (or join) keys based on overlapping names:
"""
data = pd.merge(pd.merge(ratings, users), movies)
data.ix[0]

# To get mean movie ratings for each film grouped by gender:
mean_ratings = data.pivot_table('rating', index='title', columns='gender', aggfunc='mean')

# Filter down to movies that received at least 250 ratings:
ratings_by_title = data.groupby('title').size()
ratings_by_title[:10]
active_titles = ratings_by_title.index[ratings_by_title >= 250]
mean_ratings = mean_ratings.ix[active_titles]

# To see the top films among female viewers, we can sort by the F column in descending order:
top_female_ratings = mean_ratings.sort_index(by='F', ascending=False)

# Find the movies that are most divisive between male and female viewers:
mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_index(by='diff')
sorted_by_diff[:15]
sorted_by_diff[::-1][:15]

# Standard deviation of rating grouped by title:
rating_std_by_title = data.groupby('title')['rating'].std()

# Filter down to active_titles:
rating_std_by_title = rating_std_by_title.ix[active_titles]

# Order Series by value in descending order:
rating_std_by_title.sort_values(ascending=False)[:10]

# Order Series by index in descending order:
rating_std_by_title.sort_index(ascending=False)[:10]