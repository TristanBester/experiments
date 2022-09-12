from pymongo import MongoClient


def create_client():
    client = MongoClient("mongodb://root:example@localhost:27017/")
    return client


def log_result(db_name, dataset_name, info):
    client = create_client()

    db = client[db_name]
    col = db[dataset_name]

    col.insert_one(info)

