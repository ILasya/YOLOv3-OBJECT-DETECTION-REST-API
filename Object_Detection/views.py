from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt


import io
from PIL import Image
import cv2
import numpy as np
from base64 import b64decode
from .utils import *
from .Darknet import DarkNet


@csrf_exempt
def object_detection_api(api_request):
    json_object = {'success': False}

    if api_request.method == "POST":

        if api_request.POST.get("image64", None) is not None:
            base64_data = api_request.POST.get("image64", None).split(',', 1)[1]
            data = b64decode(base64_data)
            data = np.array(Image.open(io.BytesIO(data)))
            result, detection_time = detection(data)

        elif api_request.FILES.get("image", None) is not None:
            image_api_request = api_request.FILES["image"]
            image_bytes = image_api_request.read()
            image = Image.open(io.BytesIO(image_bytes))
            result, detection_time = detection(image, web=False)

    if result:
        json_object['success'] = True
    json_object['time'] = str(round(detection_time))+" seconds"
    json_object['objects'] = result
    print(json_object)
    return JsonResponse(json_object)


def detect_request(api_request):
    return render(api_request, 'index.html')


def detection(original_image, web=True):
    cfg_file = './yolov3_Tuned.cfg'
    weight_file = './yolov3.weights'
    names = './coco.names'

    m = DarkNet(cfg_file)
    m.load_weights(weight_file)
    class_names = load_coco_names(names)

    resized_image = cv2.resize(np.float32(original_image), (m.width, m.height))
    nms_thresh = 0.018
    iou_thresh = 0.2

    boxes, detection_time = detect_objects(m, resized_image, iou_thresh, nms_thresh)
    objects = label_objects(boxes, class_names)

    if web:
        plot_object_boxes(original_image, boxes, class_names)

    return objects, detection_time
