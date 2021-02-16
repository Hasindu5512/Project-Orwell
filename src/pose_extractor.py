import dlib
import cv2


def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


class PoseExtractor:

    def __init__(self, file_name, number_of_landmarks):
        self.number_of_landmarks = number_of_landmarks
        self.hog_face_detector = dlib.get_frontal_face_detector()
        self.face_landmark = dlib.shape_predictor(file_name)

    def get_pose(self, image):
        processed_image = process_image(image)
        faces = self.hog_face_detector(processed_image)
        points = []

        for face in faces:
            landmark = self.face_landmark(processed_image, face)
            for i in range(0, self.number_of_landmarks):
                points.append((landmark.part(i).x, landmark.part(i).y))

        return points
