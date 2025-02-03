import redis        # pip install redis
import io;

ip="34.152.38.228"
r = redis.Redis(host=ip, port=6379, db=0,password='sofe4630u')

value=r.get('OntarioTech');

with open("./recieved.jpg", "wb") as f:
    f.write(value);
    
print('Image recieved, check ./recieved.jpg')
