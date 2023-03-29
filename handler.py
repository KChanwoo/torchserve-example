"""
Reference: https://gist.github.com/jarutis/f57a3db7b4c37b59163a2ff5d8c8d54e
Custom TorchServe model handler for YOLOv8 models.
+ add code to resize the boxes to origin size of images
"""
from ts.torch_handler.base_handler import BaseHandler
import numpy as np
import base64
import torch
import torchvision.transforms as tf
import io
from PIL import Image
import cv2


class ModelHandler(BaseHandler):
    """
    Model handler for YoloV8 bounding box model
    """

    img_size = 640
    """Image size (px). Images will be resized to this resolution before inference.
    """

    def __init__(self):
        # call superclass initializer
        super().__init__()

    def preprocess(self, data):
        """Converts input images to float tensors.
        Args:
            data (List): Input data from the request in the form of a list of image tensors.
        Returns:
            Tensor: single Tensor of shape [BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE]
        """
        images = []
        origin_size = []

        transform = tf.Compose([
            tf.ToTensor(),
            tf.Resize((self.img_size, self.img_size))
        ])

        # handle if images are given in base64, etc.
        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            # force convert to tensor
            # and resize to [img_size, img_size]
            origin_size.append(image.size)
            image = transform(image)

            images.append(image)

        # convert list of equal-size tensors to single stacked tensor
        # has shape BATCH_SIZE x 3 x IMG_SIZE x IMG_SIZE
        images_tensor = torch.stack(images).to(self.device)

        return images_tensor, origin_size

    def postprocess(self, inference_output):
        inference, origin_size = inference_output
        results = []

        for idx in range(len(inference)):
            w, h = origin_size[idx]
            outputs = np.array([cv2.transpose(inference[idx].numpy())])
            rows = outputs.shape[1]

            boxes = []
            scores = []
            class_ids = []
            ratio_x = self.img_size / w
            ratio_y = self.img_size / h

            for i in range(rows):
                classes_scores = outputs[0][i][4:]
                (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
                if maxScore >= 0.25:
                    box = [
                        outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                        outputs[0][i][2], outputs[0][i][3]]
                    box[0] *= ratio_x
                    box[1] *= ratio_y
                    box[2] *= ratio_x
                    box[3] *= ratio_y
                    boxes.append(box)
                    scores.append(maxScore)
                    class_ids.append(maxClassIndex)

            result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

            detections = []
            for i in range(len(result_boxes)):
                index = result_boxes[i]
                box = boxes[index]
                class_id = class_ids[index]
                detection = {
                    'class_id': class_id,
                    'class_name': self.mapping[str(class_id)],
                    'confidence': scores[index],
                    'box': [c.item() for c in box],
                }
                detections.append(detection)

            results.append(detections)

        # format each detection
        return results

    def inference(self, data, *args, **kwargs):
        """
        The Inference Function is used to make a prediction call on the given input request.
        The user needs to override the inference function to customize it.

        Args:
            data (Torch Tensor): A Torch Tensor is passed to make the Inference Request.
            The shape should match the model input shape.

        Returns:
            Torch Tensor : The Predicted Torch Tensor is returned in this function.
        """
        image, origin_size = data
        with torch.no_grad():
            marshalled_data = image.to(self.device)
            results = self.model(marshalled_data, *args, **kwargs)
        return results, origin_size
