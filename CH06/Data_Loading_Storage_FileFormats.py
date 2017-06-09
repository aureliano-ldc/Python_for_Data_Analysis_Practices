import pandas as pd
from pandas import Series, DataFrame


# read a small comma-separated (CSV) text file:
df = pd.read_csv('ch06/ex1.csv')
df

# We could also have used  read_table and specifying the delimiter:
pd.read_table('ch06/ex1.csv', sep=',')

# Consider a file without header
# Your can allow pandas to assign default column names, or you can specify names yourself
pd.read_csv('ch06/ex2.csv', header=None)
pd.read_csv('ch06/ex2.csv', names=['a', 'b', 'c', 'd', 'message'])

# Specify the index column
names = ['a', 'b', 'c', 'd', 'message']
pd.read_csv('ch06/ex2.csv', names=names, index_col='message')

# Form a hierarchical index from multiple columns, 
# just pass a list of column numbers or names:
parsed = pd.read_csv('ch06/csv_mindex.csv', index_col=['key1', 'key2'])

# Read a table without a fixed delimiter,
# pass a regular expression as delimiter for read_table
list(open('ch06/ex3.txt'))
result = pd.read_table('ch06/ex3.txt', sep='\s+')

# skip the first, third, and fourth rows of a file with skiprows:
list(open('ch06/ex4.csv'))
pd.read_csv('ch06/ex4.csv', skiprows=[0, 2, 3])

# Handling missing values:
# The na_values option can take either a list or set of strings to consider missing values:
result = pd.read_csv('ch06/ex5.csv')
pd.isnull(result)
result = pd.read_csv('ch06/ex5.csv', na_values=['NULL'])

# Different NA sentinels can be specified for each column in a dict:
sentinels = {'message':['foo', 'NA'], 'something':['two']}
pd.read_csv('ch06/ex5.csv', na_values=sentinels)

# Only read out a small number of rows (avoiding reading the entire file)
pd.read_csv('ch06/ex6.csv', nrows=5)

# To read out a file in pieces, specify a chunksize as a number of rows:
chunker = pd.read_csv('ch06/ex6.csv', chunksize=1000)
chunker

# The TextParser object returned by read_csv allows you to iterate over the parts of the
# file according to the chunksize 
tot = Series([])
for piece in chunker:
	tot = tot.add(piece['key'].value_counts(), fill_value=0)
tot = tot.sort_values(ascending=False)


# example : value_counts
t = ['A', 'A', 'B', 'B', 'B']
tt = Series(t)
tt.value_counts

# example: Series.add
list1 = {'A':1, 'B':2, 'C':3}
s = Series(list1)
s = s.add(tt.value_counts(), fill_value=0)

# Writing Data Out to Text Format
data = pd.read_csv('ch06/ex5.csv')

data.to_csv('ch06/out.csv')

data.to_csv(sys.stdout, sep='|') # writing to sys.stdout so it just prints the text result

data.to_csv(sys.stdout, na_rep='NULL')

data.to_csv(sys.stdout, index=False, header=False)

data.to_csv(sys.stdout, index=False, cols=['a', 'b', 'c']) # write only a subset of the columns

# For any file with a single-character delimiter, you can use Python’s built-in csv module.
import csv
f = open('ch06/ex7.csv')
reader = csv.reader(f)

for line in reader:
	print(line)

# Then to put the data in the form that you need it:
lines = list(csv.reader(open('ch06/ex7.csv')))
header, values = lines[0], lines[1:]
data_dict = {h: v for h, v in zip(header, zip(*values))}


"""JSON Data"""

obj = """
{"name": "Wes",
"places_lived": ["United States", "Spain", "Germany"],
"pet": null,
"siblings": [{"name": "Scott", "age": 25, "pet": "Zuko"},
			 {"name": "Katie", "age": 33, "pet": "Cisco"}]
}
"""

import json
result = json.loads(obj)
type(result)

# json.dumps on the other hand converts a Python object back to JSON:
asjson = json.dumps(result)
type(asjson) # All of the keys in an object must be strings.

# Convert a JSON object or list of objects to a DataFram
siblings = DataFrame(result['siblings'], columns=['name', 'age'])


""" XML and HTML: Web Scraping 
------------------------------
 this was the case with Yahoo! Finance’s stock options data:
 options are derivative contracts giving you the right to buy (call option) 
 or sell (put option) a company’s stock at some particular price (the strike) 
 between now and some fixed point in the future (the expiry). People trade both
 call and put options across many strikes and expiries; 
 this data can all be found together in tables on Yahoo! Finance.
"""

from lxml.html import parse
from urllib.request import urlopen

# find the URL you want to extract data from, open it and parse the stream:
parsed = parse(urlopen('http://finance.yahoo.com/q/op?s=AAPL+Options'))
doc = parsed.getroot()

# get a list of every URL linked to in the document
# links are  a tags in HTML
# Using the document root’s findall method along with an XPath
links = doc.findall('.//a')
links[15:20]

# to get the URL and link text you have to use each element’s get method (for the URL) 
# and text_content method (for the display text):
lnk = links[28]
lnk.get('href')
lnk.text_content()

# getting a list of all URLs in the document 
urls = [lnk.get('href') for lnk in doc.findall('.//a')]
urls[-10:]

# these were the two tables containing the call data and put data
tables = doc.findall('.//table')
calls = tables[1]
puts = tables[2]

# Each table has a header row followed by each of the data rows:
rows = calls.findall('.//tr')

# For the header as well as the data rows, we want to extract the text from each cell; 
# in the case of the header these are  th cells and td cells for the data:
def _unpack(row, kind='td'):
	elts = row.findall('.//%s' % kind)
	return [val.text_content() for val in elts]

_unpack(rows[0], kind='th')
_unpack(rows[1], kind='td')

# Combining all of these steps together to convert this data into a DataFrame
from pandas.io.parsers import TextParser

def parse_options_data(table):
	rows = table.findall('.//tr')
	header = _unpack(rows[0], kind='th')
	data = [_unpack(r) for r in rows[1:]]
	return TextParser(data, names=header).get_chunk()

call_data = parse_options_data(calls)
put_data = parse_options_data(puts)

""" Parsing XML with lxml.objectify """
# Using lxml.objectify, we parse the file and get a reference to the root node of the XML file with getroot 
from lxml import objectify

path = 'Performance_MNR.xml'
parsed = objectify.parse(open(path))
root = parsed.getroot()
# root.INDICATOR return a generator yielding each  <INDICATOR> XML element
root

# For each record, we can populate a dict of tag names to data values:
data = []
skip_fields = ['PARENT_SEQ', 'INDICATOR_SEQ',
				'DESIRED_CHANGE', 'DECIMAL_PLACES']

for elt in root.INDICATOR:
	el_data = {}
	for child in elt.getchildren():
		if child.tag in skip_fields:
			continue
		el_data[child.tag] = child.pyval
	data.append(el_data) # unsuccessful

# Lastly, convert this list of dicts into a DataFrame:
perf = DataFrame(data)
perf


""" Binary Data Formats"""
# One of the easiest ways to store data efficiently in binary format is using Python’s built-in pickle serialization.
frame = pd.read_csv('ch06/ex1.csv')
frame
frame.to_pickle('ch06/frame_pickle') # have a look at the file

pd.read_pickle('ch06/frame_pickle')





