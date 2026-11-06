import os

import databento as db
from dotenv import load_dotenv

load_dotenv()
client = db.Historical(os.environ["DATABENTO_API_KEY"])

cost = client.metadata.get_cost(
    dataset="GLBX.MDP3",
    symbols=["NG.v.0"],
    stype_in="continuous",
    schema="ohlcv-1m",
    start="2020-01-01",
    end="2026-11-01",
)
print(cost)
