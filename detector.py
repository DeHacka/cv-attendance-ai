"""
detector.py — The brain of the attendance system.

KEY CONCEPTS:

1. Face Encoding (Embedding)
   face_recognition.face_encodings() returns a 128-float numpy array.
   This is the "fingerprint" of a face. Similar faces → similar vectors.

2. Distance Measurement
   We use numpy to compute Euclidean distance between two 128-d vectors.
   face_recognition.compare_faces() is just: distance < tolerance (default 0.6).
   We expose the raw distance so the caller can decide their own threshold.

   Distance < 0.4  → High confidence match
   Distance 0.4–0.6 → Reasonable match
   Distance > 0.6  → Likely a different person

3. Multi-photo averaging
   For each enrolled person we may have many embeddings (one per photo).
   We compare the candidate against ALL of them and take the minimum distance.
   More photos enrolled → more robust matching, especially across lighting/angles.

4. HOG vs CNN detection model
   - HOG (default): Fast, CPU-friendly. Good for clear frontal faces.
   - CNN: More accurate, handles angles/occlusion better. Needs GPU for speed.
   We default to HOG here. Switch model="cnn" when you have a GPU server.
"""

import io
import numpy as np
from PIL import Image
import face_recognition

from storage import load_all_embeddings

# Threshold: distance below this are considered a match.
# 0.5 is stricter than the default 0.6 - reduces false positives.
MATCH_THRESHOLD = 0.5

def image_bytes_to_rgb_array(image_bytes: bytes) -> np.ndarray:
   """
   Convert raw image bytes (JPEG/PNG/etc) -> numpy RGB array.
   face_recognition expects RGB, but PIL opens as RGB by default.
   OpenCV would give BGR - that's a common bug when mixing libraries.
   """

   # Take image bytes and convert to RGB
   img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
   return np.array(img)

def detect_and_embed(image_bytes: bytes) -> list[list[float]]:
   """
   Given raw image bytes, find all faces and return their embeddings.

   Returns a list because an image can contain multiple faces. Each embedding is a list of 128 floats.

   Returns an empty list if no face is found.
   """

   rgb_array = image_bytes_to_rgb_array(image_bytes)

   # Step 1: Detect face bounding boxes using HOG
   # Returns list of (top, right, bottom, left) tuples - one per face found
   face_locations = face_recognition.face_locations(rgb_array, model="hog")

   if not face_locations:
      return []

   # Step 2: For each located face, compute the 128-d embedding
   # This internally does:
   #   a) 68-point landmark detection
   #   b) Affine transform to align face (rotation/scale)
   #   c) Forward pass through ResNet → 128-float output
   face_encodings = face_recognition.face_encodings(rgb_array, face_locations)

   # Convert numpy arrays to plain Python lists for JSON serialisation
   return [encoding.tolist() for encoding in face_encodings]


def match_embedding(candidate_embedding: list[float]) -> dict:
   """
   Compare a candidate embedding against all enrolled people.

   Returns:
     {
       "matched": True,
       "person_id": "togbe",
       "name": "Togbe Sako",
       "distance": 0.38,
       "confidence": 0.62   <- 1 - distance, for human-readable display
     }

   Or if no match:
     {
       "matched": False,
       "distance": 0.72,
       "confidence": 0.28
     }
   """

   enrolled = load_all_embeddings()

   if not enrolled:
      return {"matched": False, "reason": "No enrolled faces in the system"}

   candidate = np.array(candidate_embedding)
   best_match = None
   best_distance = float("inf")

   for person in enrolled:
      # Compare candidate against every stored embedding for this person
      stored_embeddings = [np.array(e) for e in person["embeddings"]]

      # Euclidean distance: sqrt(sum((a - b)^2)) for each dimension
      # face_recognition.face_distance() is a vectorised version of this
      distances = face_recognition.face_distance(stored_embeddings, candidate)

      # Take the closest match for this person
      min_distance = float(np.min(distances))

      if min_distance < best_distance:
         best_distance = min_distance
         best_match = person

   confidence = round(1 - best_distance, 4)

   if best_distance <= MATCH_THRESHOLD:
      return {
         "matched": True,
         "person_id": best_match["id"],
         "name": best_match["name"],
         "distance": round(best_distance, 4),
         "confidence": confidence
      }
   else:
      return {
         "matched": False,
         "distance": round(best_distance, 4),
         "confidence": confidence,
         "closest_person": best_match["name"] if best_match else None,
      }


def recognize_faces_in_image(image_bytes: bytes) -> list[dict]:
   """
   Full pipeline: image → detect all faces → match each one.

   Returns a list of results, one per face found in the image.
   This is what POST /recognize calls.
   """

   rgb_array = image_bytes_to_rgb_array(image_bytes)

   face_locations = face_recognition.face_locations(rgb_array, model="hog")
   if not face_locations:
      return []

   face_encodings = face_recognition.face_encodings(rgb_array, face_locations)

   results = []

   for encoding, location in zip(face_encodings, face_locations):
      match = match_embedding(encoding.tolist())  # fixed: was match_embeddings (plural)

      # Include the bounding box so the frontend can draw it on the image
      top, right, bottom, left = location
      match["bounding_box"] = {                   # fixed: was "bouding_box" (typo)
         "top": top, "right": right, "bottom": bottom, "left": left
      }
      results.append(match)
   return results