# API

REST API for automatic tire crack detection and segmentation using Computer Vision.

## What it does
- Receives a tire image via HTTP POST
- Classifies it as **cracked / not cracked**
- If cracked: returns the original image with segmentation masks 
  highlighting crack zones by type

## Tech stack
Python · FastAPI · OpenCV · NumPy · Deep Learning

## Architecture
Mobile App → POST /analyze → FastAPI → CV Model → Segmented image response

## Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /analyze | Upload image, get classification + masks |
| GET | /health | API status |

## Results
Case 0:
<img width="2132" height="50" alt="image" src="https://github.com/user-attachments/assets/49b4efae-64b7-43e2-b645-1a722de4b0f7" />

Note: The pipeline execution is skipped if no cracks are detected. Consequently, no segmentation images or artifacts will be generated or displayed for this run.
Case 1:
<img width="3056" height="1760" alt="image" src="https://github.com/user-attachments/assets/b2a2da36-11c7-440f-b6c7-90b2794b71bc" />

<img width="2158" height="44" alt="image" src="https://github.com/user-attachments/assets/9fa47437-9789-49e1-bfa5-706b1ab1df6a" />


# Author
CARLOS GADIEL PERALTA PEÑA
