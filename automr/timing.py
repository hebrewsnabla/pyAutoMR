from time import time, process_time
from functools import wraps

def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        t0, p0 = time(), process_time()
        result = f(*args, **kwargs)
        t1, p1 = time(), process_time()
        print("time for {0:20s},  Wall: {1:5.3f} s,  CPU: {2:5.3f} s,  ratio {3:5.1f}%"
                  .format(f.__qualname__, t1-t0, p1-p0, (p1-p0)/(t1-t0) * 100))
        return result
    return wrapper