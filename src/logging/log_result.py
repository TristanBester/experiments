from pymongo import MongoClient


def create_client():
    client = MongoClient(
        "mongodb+srv://trist:nicepassword@dtc-cluster.kff2aq7.mongodb.net/test"
    )
    return client


def create_experiment(exp_name, hparams):
    client = create_client()

    db = client["experiments"]
    col = db[exp_name]

    res = col.insert_one(hparams)
    return res.inserted_id


def log_result(model_name, result):
    client = create_client()

    db = client["results"]
    col = db[model_name]
    col.insert_one(result)

