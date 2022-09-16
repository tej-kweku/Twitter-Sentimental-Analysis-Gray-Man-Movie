#!/usr/bin/python
import datetime as dt
import time

from scheduler import Scheduler
import scheduler.trigger as trigger


def work():
    print(dt.datetime.now().isoformat())


if __name__ == "__main__":
    work()
    
    schedule = Scheduler()
    schedule.cyclic(dt.timedelta(seconds=10), work) 
        
    while True:
        schedule.exec_jobs()
        time.sleep(1)
