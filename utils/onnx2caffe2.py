import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend

# onnx 의 ModelProto를 불러옵니다. model은 protobuf 오브젝트입니다.
model = onnx.load("super_resolution.onnx")

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
