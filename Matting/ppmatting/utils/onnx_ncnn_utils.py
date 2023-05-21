import numpy as np
import ncnn
import onnxruntime
import onnx

def ncnnGetOut(filePath, data, outName):
    net = ncnn.Net()
    net.load_param(filePath + "-sim-opt.param")
    net.load_model(filePath + "-sim-opt.bin")
    mat_np  = data[0]
    mat_in = ncnn.Mat(mat_np)
    ex = net.create_extractor()
    ex.input("x", mat_in)
    ret, mat_out = ex.extract(outName)
    mat_out_np = np.array(mat_out)
    return np.expand_dims(mat_out_np, 0)

def ncnnGetOutShape(filePath, data, outName):
    print(outName + ":", ncnnGetOut(filePath, data, outName).shape)

def onnxGetOut(filePath, data, outName):
    model = onnx.load(filePath + '.onnx')
    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    ort_sess = onnxruntime.InferenceSession(model.SerializeToString())
    ort_inputs = {ort_sess.get_inputs()[0].name: data}
    if type(outName) is str:
        ort_outs = ort_sess.run([outName], ort_inputs)
        return ort_outs[0]
    else:
        ort_outs = ort_sess.run(outName, ort_inputs)
        ort_outs_map = {}
        for i, x in enumerate(ort_outs):
            ort_outs_map[outName[i]] = x
        return ort_outs_map

def compareNcnnOnnx(inData, pathPrefix = "params/modnet"):
    cmpTokens = open(pathPrefix + '-compare.list', 'r').read().splitlines()
    onnxOuts = onnxGetOut(pathPrefix, inData, [ x if ":" not in x else x.split(":")[1] for x in cmpTokens])
    for i, token in enumerate(cmpTokens):
        print(f"[{i}/{len(cmpTokens)}]Compare " + token)
        if ":" not in token:
             token = token + ":" + token
        token = token.split(":")
        ncnnOut = ncnnGetOut(pathPrefix, inData, token[0])
        np.testing.assert_allclose(ncnnOut, onnxOuts[token[1]], rtol=1.0, atol=1e-05)
    print("The difference of results between ONNXRuntime and NCNN looks good!")