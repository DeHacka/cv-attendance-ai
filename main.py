"""
main.py — FastAPI service for CV-based attendance.

Endpoints:
  POST /enroll         → Register a person's face
  POST /recognize      → Identify faces in an image
  GET  /enrolled       → List all registered people
  DELETE /enrolled/{id} → Remove a person

How your Node.js backend calls this:
  const formData = new FormData();
  formData.append("name", "Togbe Sako");
  formData.append("person_id", "togbe_sako");
  formData.append("photo", fs.createReadStream("togbe.jpg"));
  await axios.post("http://localhost:8001/enroll", formData, {
    headers: formData.getHeaders()
  });

Run with:
  uvicorn main:app --host 0.0.0.0 --port 8001 --reload
"""

import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from detector import detect_and_embed, recognize_faces_in_image
from storage import save_embedding, list_enrolled, delete_person

app = FastAPI(
    title="CV Attendance AI Service",
    description="Face recoginition microservice. Pair with your Node.js backend.",
    version="1.0.0",
)

# Allow your Node.js backend to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    """Quick ping to confirm service is alive. Use with cron-job.org keepalive."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/enroll")
async def enroll_person(
    name: str = Form(..., description="Full name of the person"),
    person_id: str = Form(None, description="Optional stable ID (e.g. student ID). Auto-generated if not provided."),
    photo: UploadFile = File(..., description="Photo of the person's face"),
):
    """
    Register a person in the attendance system.

    Best practice: enroll 3–5 photos per person in different lighting/angles.
    Call this endpoint once per photo. The system averages them at match time.

    The photo is not stored — only the 128-d embedding vector is saved.
    This is important for privacy: you can't reconstruct a face from embeddings.
    """
    image_bytes = await photo.read()
    embeddings = detect_and_embed(image_bytes)

    if not embeddings:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the uploaded photo. Use a clear, well-lit frontal image."
        )

    if len(embeddings) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"Multiple faces detected ({len(embeddings)}). Please upload a photo with only one person."
        )

    # Generate a stable ID if not provided
    pid = person_id or str(uuid.uuid4())

    # Store the single embedding
    save_embedding(person_id=pid, name=name, embedding=embeddings[0])

    return {
        "success": True,
        "person_id": pid,
        "name": name,
        "message": f"Face enrolled for {name}. Enroll more photos for better accuracy.",
    }


@app.post("/recognize")
async def recognize(
    photo: UploadFile = File(..., description="Image to identify faces in"),
):
    """
    Identify all faces in an image.

    Returns one result object per face detected, including:
    - matched: bool
    - name: str (if matched)
    - confidence: float (0–1, higher is better)
    - bounding_box: pixel coordinates for drawing the box on screen
    - distance: raw Euclidean distance (lower = more similar)

    Tip: confidence > 0.6 is a strong match. Below 0.5 treat as uncertain.
    """
    image_bytes = await photo.read()  # fixed: was missing await (caused coroutine error)
    results = recognize_faces_in_image(image_bytes)

    if not results:
        return {
            "faces_found": 0,
            "results": [],
            "message": "No faces detected in the image.",
        }

    return {
        "faces_found": len(results),
        "results": results,
        "timestamp": datetime.utcnow().isoformat(),  # fixed: was missing () after utcnow
    }


@app.get("/enrolled")
def get_enrolled():
    """
    List all enrolled people.
    Does NOT return raw embedding vectors - just metadata.
    """
    people = list_enrolled()  # fixed: was empty, missing return statement
    return {"count": len(people), "people": people}


@app.delete("/enrolled/{person_id}")
def remove_person(person_id: str):
    """Remove a person from the attendance system."""
    deleted = delete_person(person_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Person '{person_id}' not found.")
    return {"success": True, "message": f"Person '{person_id}' removed."}


@app.post("/recognize/base64")
async def recognize_base64(payload: dict):
    """
    Alternative: send image as base64 JSON instead of multipart.
    Useful when calling from React Native without FormData.

    Body: { "image": "<base64 string>", "mime_type": "image/jpeg" }
    """
    import base64

    b64 = payload.get("image", "")
    if not b64:
        raise HTTPException(status_code=400, detail="Missing 'image' field.")

    # Strip data URI prefix if present (e.g. "data:image/jpeg;base64,...")
    if "," in b64:
        b64 = b64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding.")

    results = recognize_faces_in_image(image_bytes)

    return {
        "faces_found": len(results),
        "results": results,
        "timestamp": datetime.utcnow().isoformat(),
    }