import yaml
import os, subprocess
import cv2
import typer
import numpy as np
import time
import asyncio
from furiosa.server.model import FuriosaRTModel, FuriosaRTModelConfig
from utils.postprocess import ObjDetPostProcess
from utils.preprocess import YOLOPreProcessor, letterbox

class WarboyRunner:
    def __init__(self, model, input_path, runner_info):
        self.model = FuriosaRTModel(
            FuriosaRTModelConfig(
                name="borde",
                model= model,
                batch_size=1,
                worker_num=4,
            )
        )
        self.preprocessor = YOLOPreProcessor()
        self.postprocessor = ObjDetPostProcess("yolov5m", runner_info)
        self.input_shape = runner_info["input_shape"]
        self.result_path = os.path.join(os.getcwd(), "result")
        self.input_path = input_path
        self.img_names = os.listdir(str(input_path))

    async def load(self):
        await self.model.load()

    async def process(self, img_name):
        img_path = os.path.join(self.input_path, img_name)
        img_name = img_name.split('.')[0]
        img = cv2.imread(img_path)
        input_, preproc_params = self.preprocessor(img, new_shape=self.input_shape)
        output = await self.model.predict(input_) 
        out = self.postprocessor(output, preproc_params, img)
        status = cv2.imwrite(os.path.join(self.result_path, img_name+".bmp"), out)
        return

    async def run(self):
        await asyncio.gather(*(self.task(worker_id) for worker_id in range(2)))

    async def task(self, worker_id):
        for i, img_name in enumerate(self.img_names):
            if i % 2 == worker_id:
                await self.process(img_name)

app = typer.Typer(pretty_exceptions_show_locals=False)

def get_params_from_cfg(cfg):
    model_config = open(cfg)
    model_info = yaml.load(model_config, Loader=yaml.FullLoader)
    model_config.close()

    return model_info["model_info"], model_info["runner_info"]

async def startup(runner):
    await runner.load()

async def run_model(runner):
    await runner.run()

@app.command()
def main(cfg, input_path):
    model_info, runner_info = get_params_from_cfg(cfg)

    model_path = model_info["i8_onnx_path"]
    input_datas = os.listdir(input_path)

    runner = WarboyRunner(model_path, input_path, runner_info)
    asyncio.run(startup(runner))
    
    t1 = time.time()
    asyncio.run(run_model(runner))
    t2 = time.time()
    print(f"FPS: {len(input_datas)/(t2-t1):.2f}")

if __name__ == "__main__":
    app()
