import json

import redis

fp = "/Users/spencer.trinhkinnate.com/Documents/gitrepos/sar-view-cra/backend/test.json"

with open(fp, "r") as r:
    content = r.read()


payload = json.loads(content)
r = redis.StrictRedis("localhost", password="kinnate")
r.set("test", json.dumps(payload))

read_dict = r.get("test")
