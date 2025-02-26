import kagglehub
import pandas as pd
import sqlite3 as sql

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = kagglehub.dataset_download("dhruvildave/new-york-city-taxi-trips-2019")
    print("Path to dataset files:", path)


    jan_path = path + "\\2019\\2019-01.sqlite"
    conn = sql.connect(jan_path)
    cursor = conn.cursor()

    query = "SELECT name FROM sqlite_master WHERE type='table';"
    print("DB Tables: ", cursor.execute(query).fetchall())

    query = "SELECT * from tripdata"
    data = cursor.execute(query).fetchmany(100)
    df = pd.DataFrame.from_records(data)
    print("DB Trips: ", df)
    print("Connected to January 2019 dataset")
