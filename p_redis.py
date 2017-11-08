import redis
pool=redis.ConnectionPool(host='192.168.226.128',port=6379)
r=redis.Redis(connection_pool=pool)
f=open("e:\\name1.txt",'r')
line=f.readline()
while line:
  print line,
  r.lpush('list', line.strip())
  line = f.readline()
f.close()