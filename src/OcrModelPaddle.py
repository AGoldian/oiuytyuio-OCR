import glob

import cv2
import enchant
from paddleocr import PaddleOCR, draw_ocr


# from mmocr.utils.ocr import MMOCR
# import mmcv

class OcrModelPaddle:
    def __init__(self, prefer_lang='ru'):
        self.result_en = None
        self.result = None
        if prefer_lang == 'unknown':
            self.ocr_prefer = PaddleOCR(
                use_angle_cls=True,
                det=True,
                rec=True,
                cls=True,
                use_gpu=True,
                use_space_char=True,
            )
        else:
            self.ocr_prefer = PaddleOCR(
                use_angle_cls=True,
                lang=prefer_lang,
                det=True,
                rec=True,
                cls=True,
                use_gpu=True,
                use_space_char=True,
            )
        if prefer_lang != 'en':
            self.ocr_en = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                det=True,
                rec=True,
                cls=True,
                use_gpu=True,
                use_space_char=True,
            )

    def predict(self, img, is_output_max_prob=True):
        self.result = self.ocr_prefer.ocr(img, cls=True, det=True, rec=True)[0]
        if is_output_max_prob:
            self.result_en = self.ocr_en.ocr(img, cls=True, det=True, rec=True)[0]

        boxes = [line[0] for line in self.result]
        txts = [line[1][0] for line in self.result]
        scores = [line[1][1] for line in self.result]

        if is_output_max_prob:
            boxes_en = [line[0] for line in self.result_en]
            txts_en = [line[1][0] for line in self.result_en]
            scores_en = [line[1][1] for line in self.result_en]
            boxes_res = []
            txts_res = []
            scores_res = []
            for i in range(min(len(txts), len(txts_en))):
                if scores[i] > scores_en[i]:
                    boxes_res.append(boxes[i])
                    txts_res.append(txts[i])
                    scores_res.append(scores[i])
                else:
                    boxes_res.append(boxes_en[i])
                    txts_res.append(txts_en[i])
                    scores_res.append(scores_en[i])
            return boxes_res, txts_res, scores_res
        return boxes, txts, scores

    def visualize(self, img, font_path):
        boxes = [line[0] for line in self.result]
        txts = [line[1][0] for line in self.result]
        scores = [line[1][1] for line in self.result]
        im_show = draw_ocr(img, boxes, txts, scores, font_path='simfang.ttf')
        return im_show

    def get_score(self, string1, string2):
        return 1.0 - (enchant.utils.levenshtein(string1, string2) / max(len(string1), len(string2)))


class OcrModelMM:
    def __init__(self, recog='ABINet'):
        self.result = None
        self.mmocr = MMOCR(det='TextSnake', recog='SAR',
                           cls=True,
                           use_gpu=True,
                           )

    def predict(self, img):
        self.mmocr.readtext(img,
                            print_result=True,
                            output=r' ',
                            )


import pandas as pd
import pandas as pd
import cv2
if __name__ == '__main__':
    # mm = OcrModelMM()
    # img = cv2.imread("")
    # mm.predict(img)
    ocr = OcrModelPaddle(prefer_lang='ru')
    df = pd.read_csv(r"")
    copy_df = df.copy()
    for i in range(0, len(df)):
        img_path = df.iloc[i].image_path
        text = df.iloc[i].output
        output = {"left": 1, "top": 2, "width": 3, "height": 4, "label": "", "shape": "rectangle"}
        res = []
        k = eval(df['output'].iloc[i])[0]
        img_path = df.iloc[i].image_path
        img = cv2.imread('' + img_path)
        coord, text_b, prob = ocr.predict(img,False)
        for t in k:
            output['left'] = t['left']
            output['top'] = t['top']
            output['width'] = t['width']
            output['height'] = t['height']
            output['label'] = text_b
            output['shape'] = "rectangle"
            res.append(output)
        copy_df.output.iloc[i] = res
    copy_df.to_csv(r' ', index=False)
