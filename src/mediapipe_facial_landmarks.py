"""
This module provides a class for detecting and visualizing facial landmarks using 
MediaPipe's FaceMesh. It includes methods for processing images to detect facial 
landmarks and draw them on the image.
"""

import mediapipe as mp
import cv2
import numpy as np
import pandas as pd

class MediapipeFacialLandmarks:
    """A class for detecting and visualizing facial landmarks using MediaPipe's FaceMesh.
    """
    def __init__(self, max_num_faces = 1):
        """Initializes the FaceMesh detector.

        Args:
            max_num_faces (int, optional): Maximum number of faces to detect. 
            Defaults to 1.
        """
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh_images = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def get_facial_patterns(self, image):
        """Processes the input image to detect facial landmarks.

        Args:
            image (np.ndarray): The input BGR image (as used by OpenCV).

        Returns:
             Tuple[np.ndarray, pd.DataFrame]: A tuple containing:
                - The annotated image with facial landmarks drawn.
                - A pandas DataFrame containing the flattened landmark coordinates
                  (x, y, z) for the first detected face, or None if no face is detected.
        """
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh_images.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=(
                        self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
                )

                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=(
                        self.mp_drawing_styles.get_default_face_mesh_contours_style())
                )

                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=(
                    self.mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                )

                landmarks_array = np.array(
                    [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark]
                    ).flatten()

                # features_array = pd.DataFrame([landmarks_array])

        return image, landmarks_array, face_landmarks
    


def main():
    """Main function to run the facial landmark detection and visualization using webcam 
    input. This function initializes the MediapipeFacialLandmarks class, captures video 
    from the webcam, and processes each frame to detect and visualize facial landmarks.
    """

    face_mesh_detector = MediapipeFacialLandmarks(max_num_faces =1)

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        
        try:
            image_, landmarks_array = face_mesh_detector.get_facial_patterns(image)
        
        except Exception as e:
            print(f"Error processing image: {e}")
            continue

        cv2.imshow('MediaPipe Face Mesh', image_)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows() 

    
if __name__ == "__main__":
    main()