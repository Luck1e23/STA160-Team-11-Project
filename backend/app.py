import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import gradio as gr


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "No Finding",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]
N_CLASSES = len(LABELS)


def load_trained_resnet(checkpoint_path="nih_resnet50_finetuned_best.pth"):
    """
    Build a ResNet50 without downloading weights and load your fine tuned checkpoint.
    """
    model = resnet50(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, N_CLASSES)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model


model = load_trained_resnet()



preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                          
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)), 
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],              
        std=[0.229, 0.224, 0.225],
    ),
])



def predict(image: Image.Image):
    """
    Gradio callback. Takes a PIL image, returns label probabilities.
    """
    if image is None:
        return {}


    img = image.convert("L")
    x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    return {label: float(p) for label, p in zip(LABELS, probs)}



description_text = """
Upload a frontal chest X ray image to see the model's predicted disease probabilities.

This demo uses a ResNet50 model fine tuned on the NIH Chest X ray dataset.
Results are for education and research only and must not be used for diagnosis or treatment.
"""

custom_css = """
.gradio-container {
    background: #020617;
    color: #e5e7eb;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.hero-card {
    border-radius: 0.75rem;
    padding: 1.25rem 1.5rem;
    background: radial-gradient(circle at top left, #0ea5e9 0, #020617 55%);
    border: 1px solid #1f2937;
    color: #f9fafb;
    margin-bottom: 0.75rem;
}

.hero-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.hero-subtitle {
    font-size: 0.95rem;
    opacity: 0.9;
}

.metric-row {
    display: flex;
    gap: 0.75rem;
    margin-top: 0.75rem;
    flex-wrap: wrap;
}

.metric {
    flex: 1;
    min-width: 150px;
    padding: 0.6rem 0.75rem;
    border-radius: 0.75rem;
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid #1f2937;
    font-size: 0.8rem;
}

.metric-label {
    text-transform: uppercase;
    font-size: 0.7rem;
    letter-spacing: 0.06em;
    color: #9ca3af;
}

.metric-value {
    font-size: 0.95rem;
    font-weight: 600;
    color: #e5e7eb;
}

input, button, .gr-button {
    font-family: inherit;
}
"""

with gr.Blocks(title="Chest X ray Disease Classifier") as demo:
    gr.HTML(f"<style>{custom_css}</style>")

    gr.HTML(
        """
        <div class="hero-card">
          <div class="hero-title">Chest X ray Disease Classifier</div>
          <div class="hero-subtitle">
            Deep learning model for multi label thoracic disease prediction using ResNet50 fine tuned on the NIH Chest X ray dataset.
          </div>
          <div class="metric-row">
            <div class="metric">
              <div class="metric-label">Model</div>
              <div class="metric-value">ResNet50 transfer learning</div>
            </div>
            <div class="metric">
              <div class="metric-label">Validation macro AUC</div>
              <div class="metric-value">about 0.80</div>
            </div>
            <div class="metric">
              <div class="metric-label">Validation macro F1</div>
              <div class="metric-value">about 0.29</div>
            </div>
          </div>
        </div>
        """
    )

    gr.Markdown(
        description_text
        + "\n\nUpload a frontal chest X ray on the left. The model will output the top predicted thoracic findings and their probabilities on the right."
    )

    with gr.Row():
        # Left: image input
        with gr.Column():
            img_input = gr.Image(
                type="pil",
                label="Chest X ray image",
                height=360,
            )
            run_btn = gr.Button("Run prediction", variant="primary")

        # Right: results
        with gr.Column():
            gr.Markdown("### Predicted findings")
            preds_output = gr.Label(
                num_top_classes=5,
                label="Top predicted diseases (probabilities)",
            )
            gr.Markdown(
                """
                **How to interpret these results**

                * Values are probabilities between 0 and 1.  
                * Higher values mean higher model confidence.  
                * Several labels can be positive at the same time because this is a multi label task.  
                * Class imbalance means rare diseases can still be under detected.
                """
            )

    # Wire button to prediction
    run_btn.click(
        fn=predict,
        inputs=img_input,
        outputs=preds_output,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # set to False if you only want local access
    )
