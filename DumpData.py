from db_init import *
import csv

db_file = "var/NLP.sqlite3"
conn = create_connection(db_file)
target_query = "insert into target(id, target, comment_text) VALUES(?, ? ,?)"
total_size = 100000

with open("data/train.csv", "r") as csv_file:
    rows = csv.reader(csv_file, delimiter = ",")
    count = 0
    for row in rows:
        if (count >= 1 and count <= total_size):
            target_target = (row[0], row[1], row[2])
            insert_tasks(conn, target_query, target_target)
            # insert_tasks(conn, other_target_query, other_target_target)
            # insert_tasks(conn, hate_index_query, hate_index_target)
        count += 1