'''Spelling correction and save it into a csv file'''
import pandas as pd
import sqlite3
import pandas as pd
import enchant
from db_init import *
from fuzzywuzzy import fuzz


db_file = "var/NLP.sqlite3"
conn = create_connection(db_file)
surveys_df = pd.read_sql_query("SELECT * from target", conn)
d = enchant.Dict("en_US")

for index, row in surveys_df.iterrows():
    words = row["comment_text"].split()
    for i in range(len(words)):
        if not d.check(word[i]) and d.suggest(word[i]):
            words[i] = d.suggest(words[i])[0]
    row["comment_text"] = " ".join(words)


surveys_df.to_csv(r'/Users/xietiany/Documents/NLP/data/clean_data_1.csv', index = None, header=True)



