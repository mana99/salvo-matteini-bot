
import json
import pandas as pd
from os.path import join, dirname
from sqlalchemy import create_engine

PRJ_DIR = join(dirname(__file__), '..')
INPUT_DIR = join(PRJ_DIR, "input")
WORD_EMBEDDING_PATH = join(INPUT_DIR, 'twitter128.sqlite')
WORD_EMBEDDING_SIZE = 128


sql_engine = create_engine(f"sqlite:///{WORD_EMBEDDING_PATH}")
connection = sql_engine.raw_connection()
t128 = pd.read_sql(sql='select * from store', con=connection)

# todo lowercase keys
# t128['key_lower'] = t128['key'].apply(str.lower)

t128_dict = {w[0]: tuple(w[1:]) for w in t128.to_dict('split')['data']}

with open(join(INPUT_DIR, 'twitter128.json'), 'w') as fout:
    json.dump(t128_dict, fout)

