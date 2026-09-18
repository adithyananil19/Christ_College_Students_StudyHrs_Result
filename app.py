import gradio as gr
import pandas as pd
import joblib
from pathlib import Path

# Load model
model_path = Path(__file__).parent / "model.pkl"
model = joblib.load(model_path)


# Prediction function
def predict_result(study_hours):

    input_data = pd.DataFrame({
        "StudyHours": [study_hours]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        return "Pass"
    else:
        return "Fail"


# Gradio Interface
app = gr.Interface(
    fn=predict_result,
    inputs=gr.Number(
        label="Study Hours",
        value=0
    ),
    outputs=gr.Textbox(
        label="Predicted Result"
    ),
    title="Student Pass Predictor",
    description="Enter the number of hours studied to predict the result."
)


# Start app
app.launch(
    server_name="0.0.0.0"
    server_port=int(os.environ.get("PORT", 7860))
)
