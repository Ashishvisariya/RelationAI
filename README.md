# RelationAI

AI-powered facial relationship prediction using a Siamese Neural Network built with PyTorch and served via FastAPI.

---

## How it works

RelationAI takes two face images and predicts whether the two people are biologically related using a Siamese Network trained on face embeddings from `facenet-pytorch` (VGGFace2).

---

## Project Structure

```
RelationAI/
├── main.py                          # FastAPI app + model inference
├── model/
│   └── best_siamese_model_final.pth # Trained model weights (excluded from Git)
├── utils/                           # Preprocessing helpers
├── Frontend/                        # Static web UI
│   ├── index.html
│   ├── style.css
│   └── script.js
├── Dockerfile                       # Multi-stage production Docker image
├── .dockerignore
├── render.yaml                      # Render Blueprint (backend + frontend)
├── requirements.txt                 # Pinned Python dependencies
└── .gitignore
```

---

## Local Development

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your model weights
#    Copy best_siamese_model_final.pth → model/

# 4. Start the API
uvicorn main:app --reload

# 5. Open the frontend
#    Open Frontend/index.html in your browser
#    (or serve with: python -m http.server 3000 --directory Frontend)
```

API docs available at: http://localhost:8000/docs

---

## Deployment — Render.com

> [!IMPORTANT]
> The model `.pth` file is excluded from Git (too large). You must make it
> available inside the Docker container. See options below.

### Option A — Render Persistent Disk (recommended)

1. In your Render service → **Disks** → add a disk mounted at `/app/model`.
2. SSH into the service and upload `best_siamese_model_final.pth` once.
3. Uncomment the `disk:` block in `render.yaml`.

### Option B — Download at build time

Add a `build.sh` script that downloads the weights from cloud storage (S3, GCS, HuggingFace Hub, etc.) and save it to `model/`.

### Deploying with the Blueprint

1. Push this repo to GitHub / GitLab.
2. Go to [Render Dashboard](https://dashboard.render.com) → **New → Blueprint**.
3. Connect your repo — Render will detect `render.yaml` automatically.
4. Click **Apply** — both the API and the frontend will be deployed.

### Services created

| Service | Type | URL |
|---|---|---|
| `relationai-api` | Web Service (Docker) | `https://relationai-api.onrender.com` |
| `relationai-frontend` | Static Site | `https://relationai-frontend.onrender.com` |

> [!TIP]
> Update `API_URL` in `Frontend/script.js` to point to your deployed
> `relationai-api` Render URL before deploying the frontend.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Predict relationship from two images |

### `/predict` — Request

```bash
curl -X POST https://your-api.onrender.com/predict \
  -F "image1=@personA.jpg" \
  -F "image2=@personB.jpg"
```

### `/predict` — Response

```json
{
  "prediction": "Related",
  "confidence": 0.8732,
  "probability_related": 0.8732
}
```

---

## Tech Stack

- **Backend:** FastAPI, Uvicorn
- **Model:** PyTorch, facenet-pytorch (VGGFace2 InceptionResnetV1)
- **Frontend:** Vanilla HTML / CSS / JS
- **Deployment:** Docker, Render.com
