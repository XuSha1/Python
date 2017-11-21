import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.gen
import tornado.httpclient
import tornado.concurrent
import tornado.ioloop
import motor
import asyncio
import asyncio_redis
import time

from tornado.options import define, options

define("port", default=8000, help="run on the given port", type=int)
db=motor.MotorClient().my_database
table=db.mynum

@asyncio.coroutine
def example():
      connection=yield from asyncio_redis.Connection.create(host='localhost',port=6379)
      return connection


class SleepHandler(tornado.web.RequestHandler):
    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def get(self):
        yield tornado.gen.Task(tornado.ioloop.IOLoop.instance().add_timeout, time.time() + 5)
        self.write("when I sleep 5s<br>")
        loop = asyncio.get_event_loop()
        conn=loop.run_until_complete(example())
        loop.run_until_complete(conn.incr("totals_num"))
        t = loop.run_until_complete(conn.get("totals_num"))
        self.write("visit_num is:")
        self.write(t)


class JustNowHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("I hope just now see you<br>")
        loop = asyncio.get_event_loop()
        conn = loop.run_until_complete(example())
        total_num=loop.run_until_complete(conn.get("totals_num"))
        data= {"visit_num":total_num}
        table.insert(data)
        self.write("insert visiting number in mongo<br>")
        loop.run_until_complete(conn.set("totals_num","0"))
        self.write("number reset to 0")


if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers=[
        (r"/sleep", SleepHandler), (r"/justnow", JustNowHandler)])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()