import sqlite3
from sqlite3 import Error

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn


def select_all_tasks(conn, Query):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :param Query: String
    :return:
    """
    cur = conn.cursor()
    cur.execute(Query)
    rows = cur.fetchall()
    return rows

def select_tasks(conn, Query, target):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :param Query: String
    :param target:(a,....)
    :return:
    """
    cur = conn.cursor()
    cur.execute(Query, target)
    rows = cur.fetchall()
    return rows


def insert_tasks(conn, Query, target):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :param Query: String
    :param target: (a,b,c,d,....)
    :return:
    """
    cur = conn.cursor()
    cur.execute(Query, target)
    conn.commit()
    return cur.lastrowid


def update_tasks(conn, Query, target):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :param Query: String
    :param target: (a,b,c,d,....)
    :return:
    """
    cur = conn.cursor()
    cur.execute(Query, target)
    conn.commit()


def delete_tasks(conn, Query, target):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :param Query: String
    :param target: (a,b,c,d,....)
    :return:
    """
    cur = conn.cursor()
    cur.execute(Query, target)
    conn.commit()   
