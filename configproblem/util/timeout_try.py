import signal, time

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Timed out")

def start_timeout(t=10):
    """ Throws a timeout exception after t seconds"""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(t)

def reset_timer():
    signal.alarm(0)
