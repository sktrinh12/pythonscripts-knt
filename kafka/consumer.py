from confluent_kafka import Consumer, KafkaError

bootstrap_servers = (
    "a7c7f84bba1604e40b0f7e346ea8f520-1541441431.us-west-2.elb.amazonaws.com:9094"
)
topic = "test.kinnate"
group_id = "python-consumer-group"

consumer_config = {
    "bootstrap.servers": bootstrap_servers,
    "group.id": group_id,
    "auto.offset.reset": "earliest",
}
consumer = Consumer(consumer_config)
consumer.subscribe([topic])

try:
    while True:
        msg = consumer.poll(1.0)

        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            else:
                print(msg.error())
                break

        print(f"Received: {msg.value().decode('utf-8')}")

except KeyboardInterrupt:
    pass

finally:
    consumer.close()
