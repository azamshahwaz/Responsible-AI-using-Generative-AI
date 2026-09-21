import os
import joblib


def save_model(model):

    os.makedirs(
        "outputs/models",
        exist_ok=True
    )

    try:

        joblib.dump(
            model,
            "outputs/models/trained_model.pkl"
        )

        print(
            "\nModel Saved Successfully"
        )

    except Exception as e:

        print(
            f"\nModel Save Failed : {e}"
        )