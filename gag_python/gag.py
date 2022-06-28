import time
import cv2 as cv
import hydra

class GAGModel():
    """
    Main GAG class of predict age and gender python script.
    """

    def __init__(self):
        hydra.initialize(config_path="conf", job_name="test_app")
        cfg = hydra.compose(config_name="config")

        faceProto = cfg.model.dir + cfg.model.faceProto
        faceModel = cfg.model.dir + cfg.model.faceModel

        ageProto = cfg.model.dir + cfg.model.ageProto
        ageModel = cfg.model.dir + cfg.model.ageModel

        genderProto = cfg.model.dir + cfg.model.genderProto
        genderModel = cfg.model.dir + cfg.model.genderModel

        self.MODEL_MEAN_VALUES = cfg.predict.mean
        self.ageList = cfg.predict.ages
        self.genderList = cfg.predict.genders

        self.ageNet = cv.dnn.readNet(ageModel, ageProto)
        self.genderNet = cv.dnn.readNet(genderModel, genderProto)
        self.faceNet = cv.dnn.readNet(faceModel, faceProto)

        self.padding = 20

    def getFaceBox(self, net, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
        return frameOpencvDnn, bboxes

    def get_age_gender(self, frame, returnJson=False):
        if isinstance(frame, str): frame = cv.imread(frame)
        t = time.time()
        frameFace, bboxes = self.getFaceBox(self.faceNet, frame)
        data = []
        for bbox in bboxes:
            # print(bbox)
            face = frame[max(0,bbox[1]-self.padding):min(bbox[3]+self.padding,frame.shape[0]-1),max(0,bbox[0]-self.padding):min(bbox[2]+self.padding, frame.shape[1]-1)]

            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
            self.genderNet.setInput(blob)
            genderPreds = self.genderNet.forward()
            gender = self.genderList[int(genderPreds[0].argmax())]
            self.ageNet.setInput(blob)
            agePreds = self.ageNet.forward()
            age = self.ageList[int(agePreds[0].argmax())]

            if returnJson: 
                data.append([gender,age])
            else:
                label = "{},{}".format(gender, age)
                cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        return data if returnJson else frameFace