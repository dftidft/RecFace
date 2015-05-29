import cv2

class LandmarkData:

    img_path = 'D:/Dataset/face/Face_landmark_muct/muct-e-jpg-v1/jpg/'
    landmark_path = 'D:/Dataset/face/Face_landmark_muct/muct-landmarks/muct76-opencv.csv'

    def __init__(self):
        landmark_file = open(LandmarkData.landmark_path)
        self.lines = landmark_file.readlines()

    def get_landmark_line(self, img_name):
        for line in self.lines:
            if img_name in line:
                return line
        return None

    def convert_from_line(self, landmark_str):
        landmark_arr = landmark_str.split(',')[2:]
        landmark = []
        for i in range(len(landmark_arr) / 2):
            landmark.append((float(landmark_arr[2 * i]), float(landmark_arr[2 * i + 1])))
        return [(int(pts[0]), int(pts[1])) for pts in landmark]

    def get_landmark(self, img_name):
        line = self.get_landmark_line(img_name)
        return self.convert_from_line(line)


    def get_img(self, img_name):
        return cv2.imread(LandmarkData.img_path + img_name)

    def get_landmark_at(self, index):
        line = self.lines[index]
        return self.convert_from_line(line)

    def get_img_at(self, index):
        line = self.lines[index]
        img_name = line.split(',')[0]
        return self.get_img(img_name + '.jpg')


if __name__ == '__main__':

    landmark_data = LandmarkData()
    '''
    landmark = landmark_data.get_landmark('i000qe-fn')
    img = landmark_data.get_img('i000qe-fn.jpg')
    for pts in landmark:
        cv2.circle(img, pts, 2, (0, 0, 255), 2)
    cv2.imshow('', img)
    cv2.waitKey()
    '''

    landmark = landmark_data.get_landmark_at(150)
    img = landmark_data.get_img_at(150)
    for pts in landmark:
        cv2.circle(img, pts, 2, (0, 0, 255), 2)
    cv2.imshow('', img)
    cv2.waitKey()

