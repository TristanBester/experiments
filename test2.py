from bson.objectid import ObjectId
from pymongo import MongoClient

client = MongoClient("mongodb://root:example@localhost:27017/")


db = client["experiments"]
col = db["BeetleFly"]

id_ = col.insert_one(
    {
        "AE": {
            "seq_len": 512,
            "input_dim": 1,
            "cnn_channels": 50,
            "cnn_kernel": 10,
            "cnn_stride": 1,
            "mp_kernel": 8,
            "mp_stride": 8,
            "lstm_hidden_dim": 50,
            "deconv_kernel": 10,
            "deconv_stride": 1,
        },
        "CL": {"n_clusters": 2, "metric": "EUC"},
    }
)


db = client["results"]
col = db["mine"]

col.insert_one({"experiment_id": ObjectId(id_.inserted_id), "max_auc": 1.0})


# print(id_)

