"""
sqlite converter script from http://www.italianlp.it/we-distributed/conv_script.py
"""
import sqlite3
import sys
import codecs

outfile = codecs.open(sys.argv[2], 'w', 'utf-8')
conn = sqlite3.connect(sys.argv[1])
c = conn.cursor()

for row in c.execute("SELECT * from store"):
  if row[0] == "\t":
    continue
  outfile.write("\t".join(unicode(x) for x in row[0:-1]) + "\n")
