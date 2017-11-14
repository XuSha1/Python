#coding=utf-8
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import redis
import pymongo
import json

from tornado.options import define, options
define("port", default=8000, help="run on the given port", type=int)

r=redis.StrictRedis(host="localhost",port=6379,db=0)
client=pymongo.MongoClient(host='localhost',port=27017)
db=client.tornado_python
table=db.visit

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        greeting = self.get_argument('greeting', 'Hello')
        self.write(greeting + ', World!')
       #r.set("visit:hello_world:totals",0)
        r.incr("visit:hello_world:totals")

def reset_num(flag=False):
    while flag:
      r.set("visit:hello_world:totals",0)
      flag=False
    print("reset successful!")

def write_redis():
    t = int(r.get("visit:hello_world:totals"))
    r.rpush("visit_num", t)
    print("visiting number is:"+"%d"%t)

def write_mongo():
    tt = int(r.rpop("visit_num"))
    data = '{"visit_num":"%d"}' % tt
    data_tojson = json.loads(data)
    table.insert(data_tojson)
    print("the visiting number inserting in mongo is:"+"%d"%tt)

if __name__ == "__main__":
    write_redis()
    write_mongo()
    reset_num(True)
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers=[(r"/", IndexHandler)])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

