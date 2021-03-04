from __future__ import print_function

import time
import numpy as np
import cv2
import tensorrt as trt
import torch

import pycuda.autoinit
import pycuda.driver as cuda

from openpifpaf import headmeta
from openpifpaf.decoder import cifdet
from openpifpaf.decoder.utils.cif_seeds import CifSeeds
from .datamodule import VisPanelModule

class DummyPool():
    @staticmethod
    def starmap(f, iterable):
        return [f(*i) for i in iterable]

def _preprocess(img, input_shape, letter_box=False):
    """Preprocess an image before TRT cifdet inferencing.
    # Args
        img: int8 numpy array of shape (img_h, img_w, 3)
        input_shape: a tuple of (H, W)
        letter_box: boolean, specifies whether to keep aspect ratio and
                    create a "letterboxed" image for inference
    # Returns
        preprocessed img: float32 numpy array of shape (3, H, W)
    """
    # if letter_box:
    #     img_h, img_w, _ = img.shape
    #     new_h, new_w = input_shape[0], input_shape[1]
    #     offset_h, offset_w = 0, 0
    #     if (new_w / img_w) <= (new_h / img_h):
    #         new_h = int(img_h * new_w / img_w)
    #         offset_h = (input_shape[0] - new_h) // 2
    #     else:
    #         new_w = int(img_w * new_h / img_h)
    #         offset_w = (input_shape[1] - new_w) // 2
    #     resized = cv2.resize(img, (new_w, new_h))
    #     img = np.full((input_shape[0], input_shape[1], 3), 127, dtype=np.uint8)
    #     img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
    # else:
    img = cv2.resize(img, (input_shape[1], input_shape[0]))
    img = img.transpose((2, 0, 1)).astype(np.float32)
    return img


def _nms_boxes(detections, nms_threshold):
    """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding
    boxes with their confidence scores and return an array with the
    indexes of the bounding boxes we want to keep.
    # Args
        detections: Nx7 numpy arrays of
                    [[x, y, w, h, box_confidence, class_id, class_prob],
                     ......]
    """
    x_coord = detections[:, 0]
    y_coord = detections[:, 1]
    width = detections[:, 2]
    height = detections[:, 3]
    box_confidences = detections[:, 4] * detections[:, 6]

    areas = width * height
    ordered = box_confidences.argsort()[::-1]

    keep = list()
    while ordered.size > 0:
        # Index of the current element:
        i = ordered[0]
        keep.append(i)
        xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
        yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
        xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
        yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

        width1 = np.maximum(0.0, xx2 - xx1 + 1)
        height1 = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width1 * height1
        union = (areas[i] + areas[ordered[1:]] - intersection)
        iou = intersection / union
        indexes = np.where(iou <= nms_threshold)[0]
        ordered = ordered[indexes + 1]

    keep = np.array(keep)
    return keep

def _postprocess(trt_outputs, binding_shape):
    def apply(f, items):
        """Apply f in a nested fashion to all items that are not list or tuple."""
        if items is None:
            return None
        if isinstance(items, (list, tuple)):
            return [apply(f, i) for i in items]
        return f(items)

    print("_postprocess: binding_shape: ", binding_shape)
    total_shape = np.prod(binding_shape)
    outputs = trt_outputs[0: total_shape]
    outputs = np.reshape(outputs, tuple(binding_shape))

    outputs = torch.from_numpy(outputs)

    print("_postprocess: output shape: ", outputs.shape)

    heads = [outputs]
    heads = apply(lambda x: x.cpu().numpy(), heads)

    # index by frame (item in batch)
    head_iter = apply(iter, heads)
    heads = []
    while True:
        try:
            heads.append(apply(next, head_iter))
        except StopIteration:
            break

    return heads

def _decode(fields_batch, cifdet_decoder):
    def _mappable_annotations(fields):
        return cifdet_decoder(fields)

    worker_pool = DummyPool()
    
    result = worker_pool.starmap(
        _mappable_annotations, zip(fields_batch))

    return result

def _postprocess_preds(preds, original_im_w, original_im_h, 
    prediction_im_w, prediction_im_h):
    x_ratio = original_im_w/prediction_im_w
    y_ratio = original_im_h/prediction_im_h

    for pred in preds:
        pred.bbox[0] *= x_ratio # x
        pred.bbox[2] *= x_ratio # w
        pred.bbox[1] *= y_ratio # y
        pred.bbox[3] *= y_ratio # h

    return preds

class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple."""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    """Allocates all host/device in/out buffers required for an engine."""
    inputs = []
    outputs = []
    bindings = []
    output_idx = 0
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * \
               engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            output_idx += 1
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    """do_inference (for TensorRT 7.0+)
    This function is generalized for multiple inputs/outputs for full
    dimension networks.
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

class TrtCifdet(object):
    def _load_engine(self):
        with open(self.model_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def __init__(self, model_path, input_shape=(513,513), category_num=5, letter_box=False,
                 cuda_ctx=None):
        """Initialize TensorRT plugins, engine and context."""
        self.model_path = model_path
        self.input_shape = input_shape
        self.category_num = category_num
        self.letter_box = letter_box
        self.cuda_ctx = cuda_ctx
        if self.cuda_ctx:
            self.cuda_ctx.push()

        self.inference_fn = do_inference
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine()

        cifdet_headmeta = headmeta.CifDet('cifdet', 'vispanel_trt', VisPanelModule.categories)
        cifdet_headmeta.head_index = 0
        cifdet_headmeta.base_stride = 16
        cifdet_headmeta.upsample_stride = 2
        self.head_metas = [cifdet_headmeta,]

        self.cifdet_decoder = cifdet.CifDet(self.head_metas)

        CifSeeds.threshold = 0.5

        try:
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = \
                allocate_buffers(self.engine)
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e
        finally:
            if self.cuda_ctx:
                self.cuda_ctx.pop()

    def __del__(self):
        """Free CUDA memories."""
        del self.outputs
        del self.inputs
        del self.stream

    def detect(self, img, letter_box=None):
        """Detect objects in the input image."""
        letter_box = self.letter_box if letter_box is None else letter_box
        img_resized = _preprocess(img, self.input_shape, letter_box)

        # Set host input to the image. The do_inference() function
        # will copy the input to the GPU before executing.
        self.inputs[0].host = np.ascontiguousarray(img_resized)
        start_inference = time.time()

        if self.cuda_ctx:
            self.cuda_ctx.push()
        trt_outputs = self.inference_fn(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream)
        if self.cuda_ctx:
            self.cuda_ctx.pop()

        start_decoding = time.time()
        inference_time = start_decoding - start_inference
        print(f'inference processing time: {inference_time}')

        fields = _postprocess(trt_outputs[0], binding_shape=self.engine.get_binding_shape(1))

        result = _decode(fields, self.cifdet_decoder)

        decoding_time = time.time() - start_decoding
        print(f'decoding processing time: {decoding_time}')
        
        preds = _postprocess_preds(result[0], 
            img.shape[1], img.shape[0], 
            self.input_shape[0], self.input_shape[1])

        return preds