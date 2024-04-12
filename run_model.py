
"""
Benchmark excludes image read write time
"""
import yaml
import os, subprocess
import cv2
import typer
import numpy as np
import time
from furiosa.runtime.sync import create_runner
from utils.postprocess import ObjDetPostProcess
from utils.preprocess import YOLOPreProcessor, letterbox

app = typer.Typer(pretty_exceptions_show_locals=False)

def get_params_from_cfg(cfg):
    model_config = open(cfg)
    model_info = yaml.load(model_config, Loader=yaml.FullLoader)
    model_config.close()

    return model_info["model_info"], model_info["runner_info"]

@app.command()
def main(cfg, input_path):
    model_info, runner_info = get_params_from_cfg(cfg)
    print(f'model info: {model_info}, runner info:{runner_info}')
    
    model_name = model_info["model_name"]
    # model_path = "borde_model.enf"
    model_path = model_info["i8_onnx_path"]
    input_shape = runner_info["input_shape"]
    result_path ="result"
    if os.path.exists(result_path):
        subprocess.run(["rm","-rf", result_path])

    os.makedirs(result_path)

    pre_processor = YOLOPreProcessor()
    post_processor = ObjDetPostProcess("yolov5m", runner_info)

    input_images = os.listdir(str(input_path))
    t1, t2, t3 = 0.0, 0.0, 0.0

    with create_runner(model_path) as runner:
        for input_img in input_images:
            img_name = input_img.split('.')[0]
            
            img = cv2.imread(os.path.join(input_path, input_img))
            
            ts = time.time()
            input_, preproc_params = pre_processor(img, new_shape=input_shape)
            t1 += time.time() - ts
            
            ts = time.time()
            output = runner.run([input_])
            t2 += time.time() - ts
            
            ts = time.time()
            out = post_processor(output, preproc_params, img)
            t3 += time.time() - ts
            
            cv2.imwrite(os.path.join(result_path, input_img), out)
            
    
    print(f'Processed {len(input_images)} image')
    
    print(f"Avg Preprocessing: {1000 * t1/len(input_images):.2f}, Avg Inference: {1000 * t2/len(input_images):.2f}, Avg Postprocessing: {1000 * t3/len(input_images):.2f}")
    
    print(f"FPS: {len(input_images)/(t1 + t2 + t3):.2f}")

if __name__ == "__main__":
    app()
