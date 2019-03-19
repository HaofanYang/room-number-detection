import cv2
import numpy as np
import tensorflow as tf
# Google cloud API
from google.cloud import vision
from google.cloud.vision import types
import re
import argparse

class TOD(object):

    def __init__(self, i, o, c):
        self.videoName = i
        self.outputName = o
        self.f = open(c, 'w+')
        self.client = vision.ImageAnnotatorClient()

        self.PATH_TO_CKPT = 'frozen_inference_graph.pb'
        self.PATH_TO_LABELS = 'doorplate.pbtxt'
        self.frameCount = 1
        self.NUM_CLASSES = 5
        self.detection_graph = self._load_model()
        #self.category_index = self._load_label_map()

    def _load_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph
    """
    #----------------------------------
    # 1.如何得到一个box的index
    #   boxes[0][i]代表一frame里的一个框, 它是：
    #   (y_min, x_min, y_max, x_max) 相当于top_left 与 bottom_right 两个坐标
    # 2.如何得到一个box的score: scores[0][i]
    # 3.如何得到一个box的class: classes[0][i]
    #
    #----------------------------------
    def has_clashes(boxes, box):
        return True
    def de_duplicate(boxes, scores, classes):
        new_boxes = []
        new_scores = []
        new_classes = []
        return new_boxes, new_scores, new_classes
    """
    def detect(self, image, frame_num):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Print log to file
                final_score = np.squeeze(scores)

                for i in range (100):
                    if scores is None or final_score[i] > 0.5: 
                        im_width = image.shape[1]
                        im_height = image.shape[0]
                        y_min = int(round(boxes[0][i][0]*im_height))
                        x_min = int(round(boxes[0][i][1]*im_width))
                        y_max = int(round(boxes[0][i][2]*im_height))
                        x_max = int(round(boxes[0][i][3]*im_width))

                        # 这里是用ocr读数字的，后期用google vision api替代
                        if (classes[0][i] == 1 and scores[0][i] > 0.9):
                            roi = image[y_min : y_max, x_min : x_max]

                            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                            gray = cv2.medianBlur(gray, 3)

                            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0,255,0), 2) 
                            image_copy = vision.types.Image(content=cv2.imencode('.jpg', gray)[1].tostring())
                            response = self.client.text_detection(image=image_copy)
                            texts = ''.join([text.description for text in response.text_annotations])
                            digits = re.findall(r'\d+(?:.*\d+)?', texts)
                            if len(digits) > 0:
                                print(digits)
                                cv2.putText(image, digits[0], (x_min, y_min), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                                to_write = "{num}: [{x1}, {y1}, {x2}, {y2}], {rm}\n".format(num = frame_num, x1 = x_min, y1 = y_min, x2 = x_max, y2 = y_max, rm = digits[0])
                                self.f.write(to_write)
                cv2.imshow("frame",image)




if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_video", type=str,
    help="path to input video")
    ap.add_argument("-o", "--output_video", type=str,
    help="path to output video")
    ap.add_argument("-csv", "--csv", type=str,
    help="path to output csv file")
    args = vars(ap.parse_args())

    detector = TOD(args['input_video'], args['output_video'], args['csv'])
    
    vs = cv2.VideoCapture(detector.videoName)
    frame = vs.read()
    img = frame[1]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    size = img.shape
    out = cv2.VideoWriter(detector.outputName, fourcc, 20, (size[1], size[0]))

    r = 1300/size[0]
    dim = (int(size[1] * r), 1300)

    frame_id = 1
    detector.detect(img, frame_id)
    
    while True:
        frame_id += 1
        frame = vs.read()
        img = frame[1]
        if img is None:
            break
        detector.detect(img, frame_id)
        out.write(img)
        if cv2.waitKey(1) == 27:
            break
    
    detector.f.close()
    vs.release()
    out.release()
