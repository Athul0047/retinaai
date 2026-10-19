# RetinaAI

RetinaAI is an educational project for exploring diabetic retinopathy (DR) grading from retinal fundus photographs. The model was trained in Google Colab using the APTOS 2019 dataset and EfficientNet-B0. It is intended to predict one of five grades:

| Grade | Description |
|---:|---|
| 0 | No diabetic retinopathy |
| 1 | Mild diabetic retinopathy |
| 2 | Moderate diabetic retinopathy |
| 3 | Severe diabetic retinopathy |
| 4 | Proliferative diabetic retinopathy |

## Purpose and limitations

This project is for personal learning and educational research. It is not clinically validated, does not provide medical advice, and must not be used to diagnose, screen, or guide treatment. Model outputs are experimental predictions, not diagnoses or substitutes for assessment by a qualified eye-care professional.

## Current status

The exported configuration and weights are stored locally as `models/config.json` and `models/retinaai_model.pth`. Local model loading and prediction are implemented in `src/prediction.py`. `app.py` provides a local Streamlit image-upload interface; training data is not included and does not need to be downloaded to the Mac for inference.

## Run locally

With the project virtual environment available, install the listed runtime dependencies and start Streamlit bound to the local machine:

```sh
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

Then open the local address printed by Streamlit. The model runs locally; the app does not need Colab or the APTOS dataset at runtime.
