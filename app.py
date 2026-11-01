import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

model = load_model("garbage_resnet50.keras", compile=False)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def classify(img):
    img = img.convert("RGB").resize((384, 384))
    x = np.expand_dims(np.array(img, dtype="float32"), axis=0)
    x = preprocess_input(x)
    probs = model.predict(x, verbose=0)[0]
    return {c: float(p) for c, p in zip(class_names, probs)}

demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload a photo of one item"),
    outputs=gr.Label(num_top_classes=6, label="Prediction"),
    title="Garbage Classifier",
    description="Upload a photo of a waste item (cardboard, glass, metal, paper, plastic or trash). Works best with one item on a plain light background."
)

demo.launch()