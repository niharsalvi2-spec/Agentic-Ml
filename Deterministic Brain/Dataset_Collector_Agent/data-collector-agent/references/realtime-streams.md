# Sensors / IoT — Real-Time Streams

Continuous data from physical devices (temperature, GPS, accelerometer) needs an ingestion pipeline, not a one-shot pull.

## Kafka — high volume streams

```python
from kafka import KafkaConsumer
import json
import pandas as pd

consumer = KafkaConsumer(
    "sensor_data",
    bootstrap_servers=["localhost:9092"],
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    auto_offset_reset="earliest",
    enable_auto_commit=True,
)

records = []
for message in consumer:
    records.append(message.value)
    if len(records) >= 1000:  # batch and flush
        pd.DataFrame(records).to_csv("sensor_batch.csv", mode="a", index=False)
        records = []
```

## MQTT — lightweight IoT protocol

```python
import paho.mqtt.client as mqtt
import json

def on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode())
    print(f"Temperature: {data['temp']} | Humidity: {data['humidity']}")

client = mqtt.Client()
client.on_message = on_message
client.connect("mqtt.broker.com", 1883)
client.subscribe("home/sensors/#")
client.loop_forever()
```

## Notes

- Streaming collectors should always batch-and-flush (as above), never hold unbounded data in memory.
- Use `code-generation.md`'s logging pattern so silent connection drops are visible instead of the pipeline just going quiet.
- Kafka: use consumer groups for horizontal scaling and offset tracking so restarts don't reprocess or drop messages.
