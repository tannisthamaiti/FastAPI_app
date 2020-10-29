import uvicorn
from typing import Dict
from typing import List
from fastapi import FastAPI, Depends

from model import test
from pydantic import BaseModel

#init app
app = FastAPI(title="EarthAdaptNet Image Segmentation",
              description="Obtain semantic segmentation images (Seismic Facies) of the seismic image in input via EarthAaptNet implemented in PyTorch.")

class SegmentationRequest(BaseModel):
    Inline: int


class SegmentationResponse(BaseModel):
    Pixel_Accuracy: float
    Class_Accuracy: Dict[str, float]
    Mean_Class_Accurcy: float
    Frequency_Weighted_IoU: float
    Mean_IoU: float
    IoU: Dict[str, float]

@app.get('/get_metrics/{inline}')
def get_predict_section(inline:int):
	PA, CA, MCA, FWIoU, MIoU, IoU = test.predict_section(inline)
	return SegmentationResponse(Pixel_Accuracy = PA, Class_Accuracy = dict(zip(range(6), list(CA))), Mean_Class_Accurcy = MCA, Frequency_Weighted_IoU = FWIoU, Mean_IoU = MIoU, IoU = dict(zip(range(6), list(IoU))))

@app.post('/post_metrics', response_model=SegmentationResponse)
def predict_section(request: SegmentationRequest):
	PA, CA, MCA, FWIoU, MIoU, IoU = test.predict_section(request.Inline)
	return SegmentationResponse(Pixel_Accuracy = PA, Class_Accuracy = dict(zip(range(6), list(CA))), Mean_Class_Accurcy = MCA, Frequency_Weighted_IoU = FWIoU, Mean_IoU = MIoU, IoU = dict(zip(range(6), list(IoU))))


if __name__ == '__main__':
	uvicorn.run(app, host = "127.0.0.1", port = 8000)
