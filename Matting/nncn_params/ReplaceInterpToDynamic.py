import re

def ReplaceInterpToDynamic(ncnnParams:str, srcSize:tuple, inputName = 'x'):
    ncnnLines = open(ncnnParams, 'r').read().split('\n')
    addLayerCount = 0
    for i, nLine in enumerate(ncnnLines):
        nLine = nLine.replace(" 3=0 4=0", "")
        nLine = nLine.replace(" 1=1.000000e+00 2=1.000000e+00", "")
        interpToken = re.match(r"(?P<prefix>Interp\s+\S+\s+)(?P<inMatCnt>\d)\s+(?P<outMatCnt>\d)\s+(?P<inMat>\S+)\s+(?P<outMat>\S+)\s+(?P<paramsPrefix>.*?)3=(?P<height>\d+)\s+4=(?P<width>\d+)(?P<paramsSufffix>.*)", nLine)
        if interpToken:
            height = int(interpToken["height"])
            width = int(interpToken["width"])
            heightScale = height / srcSize[1]
            widthScale = width / srcSize[0]
            ncnnLines[i] = (f'{interpToken["prefix"]} 2 1 {interpToken["inMat"]} resize_ref_{addLayerCount} {interpToken["outMat"]} {interpToken["paramsPrefix"]} 1={heightScale} 2={widthScale} {interpToken["paramsSufffix"]}')
            addLayerCount+=1
    if addLayerCount == 0:
        print("No interp layer can be convert")
    haveInputToken = False
    for i, nLine in enumerate(ncnnLines):
        countToken = re.match(r"^(?P<layerCount>\d+)\s+(?P<matCount>\d+)$", nLine)
        inputToken = re.match(r"^(?P<prefix>Split\s+splitncnn_input0\s+\d\s+)(?P<spOutCnt>\d+)(?P<suffix>.+)$", nLine)
        if countToken:
            layerCnt, matCnt = int(countToken['layerCount']), int(countToken['matCount']) + addLayerCount
            ncnnLines[i] = f"{layerCnt} {matCnt}"
        if inputToken:
            ncnnLines[i] = f'{inputToken["prefix"]}{int(inputToken["spOutCnt"]) + addLayerCount}{inputToken["suffix"]} ' + " ".join([f"resize_ref_{x}" for x in range(addLayerCount)])
            haveInputToken = True
    if haveInputToken ==False:
        raise "!haveInputToken"
    open(ncnnParams, 'w').write('\n'.join(ncnnLines))

if __name__ == "__main__":
    import os
    import re
    nowPath = os.path.dirname(__file__)
    filePath = os.path.join(nowPath, "modnet-sim.param")
    ReplaceInterpToDynamic(filePath, (144*4, 216*4))