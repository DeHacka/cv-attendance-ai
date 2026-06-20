# detector.py
import io
import numpy as np
from PIL import Image
import face_recognition

from storage import load_all_embeddings

MATCH_THRESHOLD = 0.5
MAX_DIMENSION = 1024  # caps memory per request regardless of source photo resolution

def image_bytes_to_rgb_array(image_bytes: bytes) -> tuple[np.ndarray, float]:
   """
   Convert raw image bytes -> numpy RGB array, downscaled to MAX_DIMENSION.
   Returns (array, scale) where scale maps coordinates on the downscaled
   array back to the original image's pixel space.
   """
   img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
   original_width = img.size[0]
   img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
   scale = original_width / img.size[0]
   return np.array(img), scale


def detect_and_embed(image_bytes: bytes) -> list[list[float]]:
   rgb_array, _ = image_bytes_to_rgb_array(image_bytes)
   face_locations = face_recognition.face_locations(rgb_array, model="hog")
   if not face_locations:
      return []
   face_encodings = face_recognition.face_encodings(rgb_array, face_locations)
   return [encoding.tolist() for encoding in face_encodings]


def match_embedding(candidate_embedding: list[float]) -> dict:
   enrolled = load_all_embeddings()
   if not enrolled:
      return {"matched": False, "reason": "No enrolled faces in the system"}

   candidate = np.array(candidate_embedding)
   best_match = None
   best_distance = float("inf")

   for person in enrolled:
      stored_embeddings = [np.array(e) for e in person["embeddings"]]
      distances = face_recognition.face_distance(stored_embeddings, candidate)
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
   rgb_array, scale = image_bytes_to_rgb_array(image_bytes)
   face_locations = face_recognition.face_locations(rgb_array, model="hog")
   if not face_locations:
      return []
   face_encodings = face_recognition.face_encodings(rgb_array, face_locations)

   results = []
   for encoding, location in zip(face_encodings, face_locations):
      match = match_embedding(encoding.tolist())
      top, right, bottom, left = location
      match["bounding_box"] = {
         "top": round(top * scale), "right": round(right * scale),
         "bottom": round(bottom * scale), "left": round(left * scale)
      }
      results.append(match)
   return results