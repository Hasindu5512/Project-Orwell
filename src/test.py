import cv2
from pose_extractor import PoseExtractor


def main():
    cap = cv2.VideoCapture(0)
    extractor = PoseExtractor("landmarkFiles/shape_predictor_68_face_landmarks.dat", 68)
    while True:
        ref, frame = cap.read()
        points = extractor.get_pose(frame)
        for point in points:
            cv2.circle(frame, point, 1, (0, 255, 255), 1)
        print(points)
        cv2.imshow("Face Landmarks", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
