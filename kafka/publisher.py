from confluent_kafka import Producer
import random
import time

bootstrap_servers = (
    "a7c7f84bba1604e40b0f7e346ea8f520-1541441431.us-west-2.elb.amazonaws.com:9094"
)
topic = "test.kinnate"

producer_config = {
    "bootstrap.servers": bootstrap_servers,
    "client.id": "python-producer",
}
producer = Producer(producer_config)

for i in range(10):
    message = f"Message {i}: {random.randint(1, 100)}"
    producer.produce(topic, message.encode("utf-8"))
    print(f"Published: {message}")
    time.sleep(1)

producer.flush()
