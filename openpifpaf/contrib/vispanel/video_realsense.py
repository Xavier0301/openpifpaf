import argparse
import logging
import time
import sys
import signal

import numpy as np
import torch

import pyrealsense2 as rs

import cv2

# pylint: disable=import-error
from openpifpaf import decoder, logger, network, datasets, plugins, show, transforms, visualizer, __version__

from matplotlib import pyplot as plt  
from matplotlib import text as mpltext

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass

from PIL import Image

from . import vispanel, datamodule, trt_engine

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)
LOG.addHandler(logging.StreamHandler(sys.stdout))

cp_image_dir = "./physical_cp_dataset"
coco_image_dir = "./coco_dataset"
annotations_file = "./physical_cp_dataset/annotations.json"
out_image_dir = "./out"
out_image_dir_gt = "./out_gt"
tmp_image_dir = "./tmp"

wild_images_dir = "./wild_images"

gt_tmp_dir = "./gt_tmp/"

model_name = 'vispanel-noise-epoch80.pkl'
model_name_trt = 'eng.trt'
model_dir = '/home/xplore/openpifpaf/openpifpaf/contrib/vispanel/models/'

class Timer:
    start_t = None 

    def start(self):
        self.start_t = time.time()

    def stop(self, label):
        if self.start_t is None:
            print("Timer: start_t was None and stop() was called.")
        else:
            total_t = time.time() - self.start_t
            print(f"{label} processing time: {total_t}")
            self.start_t = None
timer = Timer()

def get_processor(device):
    model_path = model_dir + model_name
    model_cpu, _ = network.factory(checkpoint=model_path)
    model = model_cpu.to(device)

    head_metas = [hn.meta for hn in model.head_nets]
    processor = decoder.factory(
        head_metas, profile=None, profile_device=device)
    return processor, model

def get_preprocess():
    rescale_t = transforms.RescaleAbsolute(513)

    pad_t = transforms.CenterPadTight(16)

    return transforms.Compose([
        transforms.NormalizeAnnotations(),
        rescale_t,
        pad_t,
        transforms.EVAL_TRANSFORM,
    ])

def custom_data_loader(n_images, preprocess):
    train_data = vispanel.VisPanel(
        cp_image_dir=cp_image_dir,
        coco_image_dir=coco_image_dir,
        annotation_file=annotations_file,
        preprocess=preprocess,
        n_images=n_images,
        prediction=True,
        tmpDir=tmp_image_dir,
        randomSeed=512
    )

    return torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False,
        pin_memory=False, collate_fn=datasets.collate_images_anns_meta)

def get_preds(image, processor, preprocess, model, device):
    image_pil = Image.fromarray(image)
    meta = {
        'hflip': False,
        'offset': np.array([0.0, 0.0]),
        'scale': np.array([1.0, 1.0]),
        'valid_area': np.array([0.0, 0.0, image_pil.size[0], image_pil.size[1]]),
    }
    processed_image, _, meta = preprocess(image_pil, [], meta)

    preds = processor.batch(model, torch.unsqueeze(processed_image, 0), device=device)[0]

    preds = preprocess.annotations_inverse(preds, meta)
    return preds

def rs_pipeline_init():
    # Create a context object. This object owns the handles to all connected realsense devices
    print("pipeline")
    pipeline = rs.pipeline()
    print("startpipeline")

    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    pipeline.start(config)

    return pipeline

def signal_handler(sig, frame, pipeline=None):
    if pipeline is not None :
        pipeline.stop()
    plt.close('all')
    sys.exit(0)

def cli():
    plugins.register()

    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.contrib.vispanel.video_realsense',
        usage='%(prog)s [options]',
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--scale', default=1.0, help='rescale the image before prediction')
    parser.add_argument('--usetrt', action='store_true',
                        help='use TensorRT to run the model')
    parser.add_argument('--debug-depth', action='store_true',
                        help='outputs the depth')

    args = parser.parse_args()

    return args

def main():
    args = cli()

    if args.usetrt:
        model_path = model_dir + model_name_trt
        eng = trt_engine.TrtCifdet(model_path)
    else:
        device = torch.device('cuda')
        print("proc")
        processor, model = get_processor(device)
        print("preproc")
        preprocess = get_preprocess()

    print('sig')
    signal.signal(signal.SIGINT, signal_handler)

    annotation_painter = show.AnnotationPainter()
    print("subplot")
    if args.debug_depth:
        fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2)
        ax2.title.set_text("Depth map")
    else:
        fig, ax = plt.subplots()
    ax.title.set_text("Image")

    plt.axis('off')
    for ax_ in fig.axes:
        ax_.get_xaxis().set_visible(False)
        ax_.get_yaxis().set_visible(False)

    last_loop = time.time()
    
    try:
        print("pipeline")
        pipeline = rs_pipeline_init()

        print("frames")
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if args.debug_depth:
            depth_frame = frames.get_depth_frame()
    except KeyboardInterrupt:
        signal_handler(None, None, pipeline=pipeline)

    print("implot")
    implot = ax.imshow(np.asanyarray(color_frame.get_data()))
    if args.debug_depth:
        implot2 = ax2.imshow(np.asanyarray(depth_frame.get_data()), cmap='magma', vmin=0, vmax=1000) #plasma, inferno, magma

        plt.colorbar(implot2, ax=ax2, fraction=0.046, pad=0.04)

    plt.ion()
    
    frame_i = 0
    while True:
        frame_i += 1

        print('collect frame')

        timer.start()

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if args.debug_depth:
            depth_frame = frames.get_depth_frame()

        print('done collecting')

        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())
        print(image.shape)
        print(image.dtype)
        if args.debug_depth:
            depth = np.asanyarray(depth_frame.get_data())

        start = time.time()
        if args.scale != 1.0:
            image = cv2.resize(image, None, fx=args.scale, fy=args.scale)

        visualizer.Base.image(image)

        timer.stop("image collection")

        start_post = time.perf_counter()
        # preds
        print("pred")
        if args.usetrt:
            preds = eng.detect(image)
            # signal_handler(None, None, pipeline=pipeline)
            print(len(preds))
        else:
            timer.start()
            preds = get_preds(image, processor, preprocess, model, device)
            timer.stop("torch preds")

        print('imshow')

        timer.start()

        implot.set_data(image)
        if args.debug_depth:
            implot2.set_data(depth)

        for child in ax.get_children():
            if isinstance(child, mpltext.Annotation):
                child.remove()

        ax.patches = []

        annotation_painter.annotations(ax, preds)

        plt.pause(0.001)
        if frame_i == 2:
            plt.pause(300)

        timer.stop("plotting")
    
        LOG.debug('time post = %.3fs', time.perf_counter() - start_post)
        LOG.info('frame %d, loop time = %.3fs, FPS = %.3f',
                 frame_i,
                 time.time() - last_loop,
                 1.0 / (time.time() - last_loop))

        last_loop = time.time()

if __name__ == '__main__':
    main()
