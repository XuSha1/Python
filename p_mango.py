#coding=utf-8
import redis
import pymongo
import json
pool=redis.ConnectionPool(host='192.168.226.128',port=6379)
client=pymongo.MongoClient(host='192.168.226.128',port=27017)
r=redis.StrictRedis(connection_pool=pool)

db=client.geetest_python #库名
table=db['names'] #表名

for i in xrange(r.llen('list')):
  rr=r.rpop("list")
  print(rr)
  data='{"name":"%s"}'%rr
  to_json=json.loads(data)
  print(to_json)
  table.insert(to_json)
