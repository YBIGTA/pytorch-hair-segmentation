import sys
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn.init as init

import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend

from networks.fcn_mobilenetv2 import FCN

device = 'cuda' if torch.cuda.is_available() else 'cpu'



name = 'FCN_SGD_epoch_43'

model = FCN()
model.load_state_dict(torch.load(f'ckpt/{name}.pth', map_location=device)['weight'])

model.train(False)

# 샘플 X
batch_size = 2
x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

# 모델을 onnx로 변환
torch.onnx._export(model,                   # 모델
                   x,                       # 샘플 X를 넣어야 합니다. 모델의 인풋이 여러개인 경우, 텐서의 튜플로 넣을 수 있습니다.
                   f"{name}.onnx",          # 저장 경로
                   export_params=True)      # 저장할때 모델의 파라미터도 함께 저장합니다.






# onnx 의 ModelProto를 불러옵니다. model은 protobuf 오브젝트입니다.
model = onnx.load(f"{name}.onnx")

# onnx 모델을 Caffe2 API 인 NetDef 의 형태로 변환해줍니다.
prepared_backend = onnx_caffe2_backend.prepare(model)

# Caffe2 에서 실행

# Construct a map from input names to Tensor data.
# The graph of the model itself contains inputs for all weight parameters, after the input image.
# Since the weights are already embedded, we just need to pass the input image.
# Set the first input.
W = {model.graph.input[0].name: x.data.numpy()}

# Run the Caffe2 net:
c2_out = prepared_backend.run(W)[0]

# Verify the numerical correctness upto 3 decimal places
np.testing.assert_almost_equal(torch_out.data.cpu().numpy(), c2_out, decimal=3)

print("Caffe2에서 실행 -- 성공!")
