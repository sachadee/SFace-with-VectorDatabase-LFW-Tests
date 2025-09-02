import cv2
import numpy as np
from numpy import float32, int32, int64, ndarray
from typing import List,Tuple,Union, Optional, Dict, Any
import string
import time
import pybase64
from io import BytesIO
import json
import random
import os
from sdeeVectorDB import VectorDatabase

np.set_printoptions(suppress=True)

class FaceRecognitionSystem:
    """
    A facial recognition system using OpenCV DNN modules for detection and embedding,
    with a custom vector database for face storage and matching.
    Tested with lfw dataset:
        The LFW dataset contains: 12,853 images of faces collected from the web. 
        This dataset consists of the 5749 identities with 1680 people 
        ✅ Total Added: 5749 faces
            Inference Time media: 0.05
        Accuracy tested : each face tested against all face with name check.
        Result:
            Accuracy: 0,996% (12802/12853) with a threshold of 0.47 cosine
    """

    def __init__(self, 
                 detector_path='./model/face_detection_yunet_2023mar.onnx',
                 recognizer_path='./model/face_recognition_sface_2021dec.onnx',
                 db_path='vector_db',
                 collection='Sface_lfw-deepfunneledAll',
                 cropped_dir='croppedFaces',
                 target_size=320):
        """
        Initialize the face detection and recognition system.

        Args:
            detector_path (str): Path to the face detector ONNX model.
            recognizer_path (str): Path to the face recognizer ONNX model.
            db_path (str): Directory path for storing vector database.
            collection (str): Name of the vector database collection.
            cropped_dir (str): Directory to save cropped aligned face images.
            target_size (int): Size to which images are resized for detection.
        """
        
        self.detector = cv2.FaceDetectorYN.create(detector_path, "", (target_size, target_size), 0.9, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(recognizer_path, "")
        self.vecDb = VectorDatabase(data_dir=db_path, collection=collection, dim=128)
        self.target_size = target_size
        self.cropped_dir = cropped_dir
        self.threshold = 0.5

        os.makedirs(self.cropped_dir, exist_ok=True)


#####util

    def _random_id(self, length=8) -> str:
        """
        Generate a random alphanumeric ID.

        Args:
            length (int): Length of the ID string.

        Returns:
            str: Random string.
        """
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    def b642cv2 (self,b64):
        """
        Convert a base64 string to an OpenCV image.

        Args:
            b64 (str): Base64-encoded image string.

        Returns:
            np.ndarray: Decoded image or None if invalid.
        """
        if self.is_valid_base64_image(b64):
            img_bytes = pybase64.b64decode(b64)
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            return img
        return [0]

    def im2b64(self,img):
         """
         Convert an OpenCV image to a base64 string.

         Args:
             img (np.ndarray): Input image.

         Returns:
             str: Base64-encoded string.
         """
         _, img_encoded = cv2.imencode('.jpg', img)
         img_bytes = img_encoded.tobytes()
         b64 = pybase64.b64encode(img_bytes).decode('utf-8')
         return b64
    
    def is_valid_base64_image(self,base64_string):
        """
        Check if the given string is a valid base64 image.

        Args:
            base64_string (str): Input string.

        Returns:
            bool: True if valid base64 image, False otherwise.
        """
        try:
           img_data = pybase64.b64decode(base64_string)
           Image.open(BytesIO(img_data)) 
           return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def is_cv2_image(self,obj):
        """
        Check if the object is an OpenCV image (NumPy array with 2 or 3 dimensions).

        Args:
            obj: Any Python object.

        Returns:
            bool: True if it's a cv2 image.
        """
        return isinstance(obj, np.ndarray) and obj.ndim in [2, 3]

    def is_base64_string(self,s):
        """
        Determine if a string is likely a base64-encoded image.

        Args:
            s (str): Input string.

        Returns:
            bool: True if valid base64 image string.
        """

        if not isinstance(s, str) or len(s) < 20:
            return False
        if s.startswith("data:image/"):
           return True
        try:
           if "," in s:
                s = s.split(",")[1]
           pybase64.b64decode(s, validate=True)
           return True
        except Exception:
            return False

    def is_image_path(self,s):
        """
        Check if the given string is a valid image file path.

        Args:
            s (str): Path string.

        Returns:
            bool: True if file exists and is an image.
        """

        if not isinstance(s, str):
            return False
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        return os.path.isfile(s) and os.path.splitext(s)[1].lower() in valid_exts

    def load_image_as_cv2(self,input_data):
        """
        Load an image from a file path, base64 string, or already loaded image.

        Args:
            input_data (str or np.ndarray): Input source.

        Returns:
            np.ndarray: Loaded image or None if loading failed.
        """
        if self.is_cv2_image(input_data):
            return input_data

        if self.is_base64_string(input_data):
           try:
               if "," in input_data:
                    input_data = input_data.split(",")[1]
               img_data = pybase64.b64decode(input_data)
               nparr = np.frombuffer(img_data, np.uint8)
               img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
               return img
           except Exception as e:
                print("Failed to decode base64:", e)
                return None

        if self.is_image_path(input_data):
           img = cv2.imread(input_data)
           return img

        print("Unsupported input format")
        return None


#####Recognizer

    def _resize_image(self, image: np.ndarray, target_size: int = 320) -> np.ndarray:
       """
       Resize and pad image to a square of target_size x target_size while maintaining aspect ratio.

       Args:
           image (np.ndarray): Input image (HWC).
           target_size (int): Desired output size.

       Returns:
           np.ndarray: Resized and padded image.
       """
       h, w = image.shape[:2]
       scale = target_size / max(h, w)
       new_w, new_h = int(w * scale), int(h * scale)
    
       resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
       top = (target_size - new_h) // 2
       bottom = target_size - new_h - top
       left = (target_size - new_w) // 2
       right = target_size - new_w - left

       padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
       return padded

    def detect_face(self, image: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Detect the first face in the image and return an aligned cropped version.

        Args:
            image (np.ndarray): Input OpenCV image.

        Returns:
            np.ndarray: Aligned cropped face or None if no face detected.
        """

        image = self.load_image_as_cv2(image)

        if image is None:
            raise ValueError("Input image is None")

        resized = self._resize_image(image)
        self.detector.setInputSize(resized.shape[1::-1])
        
        start = time.time()
        _, faces = self.detector.detect(resized)
        if faces is None or len(faces) == 0:
            print("[WARN] No face detected.")
            return None
        aligned = self.recognizer.alignCrop(resized, faces[0])
        return aligned

    def extract_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Extract the 128D embedding vector from an aligned face.

        Args:
            aligned_face (np.ndarray): Aligned face image.

        Returns:
            np.ndarray: Embedding vector.
        """
                            
        start = time.time()
        embedding = self.recognizer.feature(aligned_face)
        return embedding[0]


    def add_face(self, image: Union[str, np.ndarray], name: str) -> Dict:
        """
        Detect,Align and embed a face from an image, then store it in the vector database.

        Args:
            image (np.ndarray): Input image.
            name (str): Person's name to associate with the embedding.

        Returns:
            dict: Status and ID of the added face.
        """

        image = self.load_image_as_cv2(image)

        aligned = self.detect_face(image)
        
        if aligned is None:
            return {"status": "failed", "reason": "No face detected"}

        embedding = self.extract_embedding(aligned)
        face_id = self._random_id()
        filename = f"{face_id}.jpg"
        filepath = os.path.join(self.cropped_dir, filename)
        cv2.imwrite(filepath, aligned)

        self.vecDb.add_vector(embedding, {
            "name": name,
            "face_id": face_id,
            "thumbnail": filename
        })

        return {"status": "added", "name": name, "uid": face_id}

    
    def del_face(self,name: str) -> Dict:
        return self.vecDb.deleteAllFiltered(where={"name": name})
        
    def search(self, query_embedding: np.ndarray, top_k=1, threshold = 0.5) -> Optional[List[Dict]]:
        """
        Search for the top-k most similar faces in the database based on cosine similarity.

        Args:
            query_embedding (np.ndarray): The embedding to match.
            top_k (int): Number of top results to return.
            threshold (float): Minimum similarity threshold.

        Returns:
            List[Dict]: Matching results or None if no match found.
        """
        results = []
        vectors = self.vecDb.get(incl=["vector", "metadata"])

        for entry in vectors:
            vec = np.array(entry["vector"], dtype=np.float32)
            score = self.recognizer.match(query_embedding, vec, cv2.FaceRecognizerSF_FR_COSINE)
            if score >= threshold:
                results.append({
                    "id": entry["metadata"].get("face_id"),
                    "name": entry["metadata"].get("name"),
                    "similarity": score,
                    "metadata": entry["metadata"]
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k] if results else None


    def match(self, image: Union[str, np.ndarray],match_name: str,threshold=0.4) -> None:
        """
        Check if the given image matches a person already in the database.

        Args:
            image (np.ndarray): Input image.
            match_name (str): Name of the person to match against.
            threshold (float): Similarity threshold.

        Returns:
            dict: Result with status, similarity, and inference time.
        """
        start = time.time()
        image = self.load_image_as_cv2(image)
        if image is None:
            print("[ERROR] No image.")
            return {"status": "error", "msg": "no image"}

        aligned = self.detect_face(image)
        if aligned is None:
            return {"status": "error", "msg": "no face detected"}


        embedding = self.extract_embedding(aligned)
        vector = []
       
        vector = self.vecDb.get(where={"name": match_name},incl=["vector", "metadata"])
        
        if len(vector) >= 1:
            vector = np.array(vector[0]["vector"], dtype=np.float32)
            score = self.recognizer.match(embedding, vector, cv2.FaceRecognizerSF_FR_COSINE)
            if score >= threshold:
                endT = time.time() - start
                
                return {"status": "success","msg": "matched", "name": match_name,"similarity": round(score, 4), "InfTime": round(endT, 4)}
            else:
                return {"status": "success","msg": "unmatched","name": match_name,"similarity": round(score,4),"threshold": threshold}
        else:
            return {"status": "error","msg": "not in database", "name": match_name}


    def recognize(self, image:Union[str, np.ndarray], k: int = 1, threshold: float = 0.5) -> None:
        """
        Recognize a person from the image by comparing with all stored faces.

        Args:
            image (np.ndarray): Input image.

        Returns:
            dict: Recognition result including name, ID, similarity, and inference time.
        """

        start = time.time()
        image = self.load_image_as_cv2(image)
        if image is None:
            return {"status": "error", "msg": "no image"}


        aligned = self.detect_face(image)
        if aligned is None:
            return {"status": "error", "msg": "no face detected"}

        embedding = self.extract_embedding(aligned)
        matches = self.vecDb.search_vectors(
                            query_vector=embedding,
                            k=k,
                            threshold=threshold
                    )
        if not matches:
            return {"status": "success","msg": "unrecognized"}
        else:
            endT = time.time() - start
            return matches
