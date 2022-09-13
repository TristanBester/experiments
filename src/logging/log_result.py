from pymongo import MongoClient


def create_client():
    client = MongoClient("mongodb://root:example@localhost:27017/")
    return client


def create_experiment(dataset_name, hparams):
    client = create_client()

    db = client["experiments"]
    col = db[dataset_name]

    res = col.insert_one(hparams)
    return res.inserted_id


def log_result(model_name, result):
    client = create_client()

    db = client["results"]
    col = db[model_name]
    col.insert_one(result)

