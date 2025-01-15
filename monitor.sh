#!/bin/bash

PROJECT_ID="sandbox-373102"

echo "Target Topic ID"
topic_id="projects/"$PROJECT_ID"/topics/tgi"
echo $topic_id

while true; do
  output=$(wget -qO- "http://localhost:8080/metrics")
  gcloud pubsub topics publish $topic_id --attribute=id="$AIP_PROJECT_NUMBER"/"$AIP_ENDPOINT_ID"/"$AIP_DEPLOYED_MODEL_ID":"$AIP_DEPLOYED_MODEL_ID" --message="$output"
  #echo "Message pushed"
  sleep 10
done