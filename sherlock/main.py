import pandas as pd
import numpy as np
import traceback
from fastapi import FastAPI, HTTPException, status

app = FastAPI()

import sherlock
from sherlock import helpers
from sherlock.deploy.model import SherlockModel
from sherlock.functional import extract_features_to_csv
from sherlock.features.paragraph_vectors import (
    initialise_pretrained_model,
    initialise_nltk,
)
from sherlock.features.preprocessing import (
    extract_features,
    convert_string_lists_to_lists,
    prepare_feature_extraction,
    load_parquet_values,
)
from sherlock.features.word_embeddings import initialise_word_embeddings
from sherlock.schema import TableData, SherlockTagsRequest, SherlockTagsResponse


@app.get("/health")
async def health():
    print("Sherlock is Up and Running!")
    return {"status": "ok"}


@app.post("/sherlock/get_tags", response_model=SherlockTagsResponse)
async def sherlock_get_tags(body: SherlockTagsRequest):
    try:

        model = SherlockModel()
        model.initialize_model_from_json(with_weights=True, model_id="sherlock")

        values = []
        for table_data in body.data:
            sample_data = table_data.sampleData
            sample_data_str = list(map(str, sample_data))
            values.append(sample_data_str)

        data = pd.Series(
            values,
            name="values",
        )

        extract_features("../temporary.csv", data)
        feature_vectors = pd.read_csv("../temporary.csv", dtype=np.float32)

        predicted_labels = list(model.predict(feature_vectors, "sherlock"))

        print(f"predicted_labels: {predicted_labels}")

        tags_response = {}
        for idx in range(len(predicted_labels)):
            tags_response[body.data[idx].columnName] = [predicted_labels[idx]]

        print(f"tags_response: {tags_response}")

        response = SherlockTagsResponse(tags=tags_response)
        return response
    except Exception as ex:
        print(f"Error: {ex}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(ex)
        )