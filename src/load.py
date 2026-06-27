import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine,text
from dotenv import load_dotenv
from tqdm.auto import tqdm
import os
import csv
import psutil
from io import StringIO
import math

# environment variables
load_dotenv(Path.cwd().parent / ".env")
db_user = os.environ["DB_USER"]
db_password = os.environ["DB_PASSWORD"]
db_name = os.environ["DB_NAME"]
db_port = os.environ["DB_PORT"]


#reading the csv
raw_data_dir = Path.cwd().parent / "data" / "raw"
df = pd.read_csv(raw_data_dir / "application_train.csv")

engine = create_engine(f"postgresql+psycopg://{db_user}:{db_password}@localhost:{db_port}/{db_name}")


# calculating the chunksize 
# for ram usage when ingesting data by analyzing the machine
n_cols = df.shape[1]
bytes_per_row = df.memory_usage(deep=True).sum() / len(df)
ram_available = psutil.virtual_memory().available
bytes_ram_limit_usage = ram_available * 0.20 #just 20% of ram
chunksize = int(bytes_ram_limit_usage / bytes_per_row)
chunksize = max(chunksize, 1_000) # def min as 1,000
chunksize = min(chunksize, 100_000) # def max as 100,000


def copy_method(table, conn, keys, data_iter):
    buf = StringIO()
    writer = csv.writer(buf) #here these method expects a file object, 
    #so even though buf is a virtualization (not a local file) it works
    writer.writerows(data_iter)
    buf.seek(0)
    cur = conn.connection.cursor()
    cur.copy(
        f"COPY {table.schema}.{table.name} ({', '.join(keys)}) FROM STDIN WITH CSV", 
        buf
    )


#we parition our dataframe by chunks, in an iterable of dataframes
def iterate_chunks(df, n_chunks):
    for i in range(0, len(df), chunksize):
        yield df[i:i+chunksize]
 
n_chunks = math.ceil(len(df) / chunksize)

with tqdm(total = len(df), desc="subiendo a postgresql (docker)", unit="filas") as pbar:
    for i, chunk in enumerate(iterate_chunks(df, n_chunks)):
        chunk.to_sql(
            name="application_train",
            schema="raw",
            con=engine,
            method=copy_method,
            if_exists="replace" if i == 0 else "append",
            index=False
        )
        pbar.update(len(chunk)) #here we dont use chunksize since the last one might be
        # shorter in length than the chunksize


