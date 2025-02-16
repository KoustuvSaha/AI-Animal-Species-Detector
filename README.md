# AI-Animal-Species-Detector

#### `README.md` – Project Documentation:

# Animal Species Classifier using ResNet50
This project uses a CNN (ResNet50) model to classify animal species (Cats, Dogs, Wildlife) from images. It includes a training module, a FastAPI endpoint, and a Gradio UI for predictions.

## Project Structure:
```
├── train_model.py      # Model training script
├── api.py              # FastAPI for predictions
├── ui.py               # Gradio interface for predictions
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

##Installation and Usage:
### 1. Clone the Repository:
```bash
git clone https://github.com/yourusername/animal-species-classifier.git
cd animal-species-classifier
```
### 2. Create a Virtual Environment and Install Dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```
### 3. Train the Model:
```bash
python train_model.py
```
### 4. Start the API Server:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
### 5. Launch the Gradio UI:
```bash
python ui.py
```

## Enhancements:
- Add GPT for species descriptions
- Enable batch predictions
- Deploy via Docker or cloud platforms

## Contributing:
Pull requests are welcome. For major changes, open an issue first to discuss what you would like to change.
```
