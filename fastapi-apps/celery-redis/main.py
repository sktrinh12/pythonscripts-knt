from celery import Celery
from fastapi import FastAPI
from os import getenv

app = FastAPI()


celery = Celery(
    __name__,
    broker=f"redis://:{getenv('REDIS_PASSWD')}@{getenv('REDIS_HOST', '127.0.0.1')}:{getenv('REDIS_PORT', '6379')}/0",
    backend=f"redis://:{getenv('REDIS_PASSWD')}@{getenv('REDIS_HOST', '127.0.0.1')}:{getenv('REDIS_PORT', '6379')}/0",
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@celery.task
def divide(x, y):
    import time

    time.sleep(5)
    return x / y
