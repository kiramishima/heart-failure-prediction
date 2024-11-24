# Heart Failure Prediction
Heart Failure Prediction is my project for Machine Learning Zoomcamp

# Problem description

# Stack

# Notebooks

# Deploy

```docker
docker run -it \
    --rm \
    --name predictor \
    -p 9696:9696 \
    -e "RUN_ID=LogisticRegression" \
    heart-api
```