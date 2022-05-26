import math
import numpy as np
import cv2
import pickle
import os


RemRadios = np.array([0.67, 1, 1.33, 1.5, 1.78])
def find_nearestIdx(a0):
    idx = np.abs(RemRadios - a0).argmin()
    return idx

def cv2ToBytes(img) -> bytes:
    imgEncodeWebp=cv2.imencode('.webp',img)[1]
    imgEncodeWebp=np.array(imgEncodeWebp)
    return imgEncodeWebp.tobytes()

def bytesToCv2(imgBuff : bytes, mode = cv2.IMREAD_COLOR) -> np.ndarray:
    image = np.asarray(bytearray(imgBuff), dtype="uint8")
    return cv2.imdecode(image, mode)

#
# meta文件数据结构：[[{image : [pos, size], mask : [pos, size], name : str}]]
#

class BgsData:
    def __init__(self, megapixel = 0.5) -> None:
        self.metaData = []
        self.pixel = megapixel * 1000000
        self.sizeList = []
        for r in RemRadios:
            height = math.sqrt(self.pixel / r)
            width = height * r
            height = round(height / 32) * 32
            width = round(width / 32) * 32
            self.sizeList.append((width, height))

    def openData(self, metaPath:str, dataPath:str = None):
        self.metaData, self.nameDict, RemRadios = pickle.load(open(metaPath, 'rb'))
        if dataPath:
            self.fileHandle = open(dataPath, 'rb+')
        else:
            self.fileHandle = None
            self.ImagesPath = os.path.split(metaPath)[0] + "Images"
            self.MasksPath = os.path.split(metaPath)[0] + "Masks"
        self.metaPath = metaPath
        pass

    def creatData(self, metaPath:str, dataPath:str = None):
        if dataPath:
            self.fileHandle = open(dataPath, 'wb')
        else:
            self.fileHandle = None
            self.ImagesPath = os.path.split(metaPath)[0] + "Images"
            self.MasksPath = os.path.split(metaPath)[0] + "Masks"
        self.metaPath = metaPath
        self.nameDict = {}
        for i in RemRadios:
            self.metaData.append([])
        pass

    def getFileData(self, pos, size):
        self.fileHandle.seek(pos)
        return self.fileHandle.read(size)

    def getData(self, radiaIndex:int, dataIndex:int) -> tuple:
        if self.fileHandle:
            imagePos, imageSize = self.metaData[radiaIndex][dataIndex]['image']
            maskPos, maskSize   = self.metaData[radiaIndex][dataIndex]['mask']
            return self.getFileData(imagePos, imageSize), self.getFileData(maskPos, maskSize)
        else:
            name = self.metaData[radiaIndex][dataIndex]['name']
            imgagePath = os.path.join(self.ImagesPath, name + ".webp")
            maskPath = os.path.join(self.MasksPath, name + ".webp")
            return open(imgagePath, 'rb').read(), open(maskPath, 'rb').read()

    def getImageData(self, radiaIndex:int, dataIndex:int, resize = True) -> tuple:
        imgBuff, mskBuff = self.getData(radiaIndex, dataIndex)
        img, mask = bytesToCv2(imgBuff), bytesToCv2(mskBuff, cv2.IMREAD_GRAYSCALE)
        if resize:
            img, mask = cv2.resize(img, self.sizeList[radiaIndex]), cv2.resize(mask, self.sizeList[radiaIndex])
        return img, mask

    def addDataMeta(self, name:str, width:int, height:int):
        fData = {}
        fData['name'] = name
        if fData['name'] in self.nameDict:
            return
        whRadio = width / height # 获取图像的宽高比
        nIdx = find_nearestIdx(whRadio)
        self.metaData[nIdx].append(fData)
        self.nameDict[fData['name']] = (nIdx, len(self.metaData[nIdx]) - 1)

    def addData(self, imagePath:str, maskPath:str):
        fData = {}
        fData['name'] = os.path.splitext(os.path.split(imagePath)[1])[0]
        if fData['name'] in self.nameDict:
            return

        img, msk = cv2.imread(imagePath, cv2.IMREAD_ANYCOLOR), cv2.imread(maskPath, cv2.IMREAD_ANYCOLOR)
        whRadio = msk.shape[1] / msk.shape[0] # 获取图像的宽高比
        #找出最接近的比例的下标, 并重新缩放到该值
        nIdx = find_nearestIdx(whRadio)
        img, msk = cv2.resize(img, self.sizeList[nIdx]), cv2.resize(msk, self.sizeList[nIdx])

        img, msk = cv2ToBytes(img), cv2ToBytes(msk)

        self.fileHandle.seek(0,2)
        fData['image'] = (self.fileHandle.tell(), len(img))
        self.fileHandle.write(img)
        fData['mask'] = (self.fileHandle.tell(), len(msk))
        self.fileHandle.write(msk)
        self.metaData[nIdx].append(fData)
        self.nameDict[fData['name']] = (nIdx, len(self.metaData[nIdx]) - 1)

    def save(self):
        pickle.dump((self.metaData, self.nameDict, RemRadios), open(self.metaPath, 'wb'))
        if self.fileHandle:
            self.fileHandle.flush()

    def getRadios(self)-> list:
        return list(RemRadios)

    def getRadiosDataSize(self, radiaIndex:int) -> int:
        return len(self.metaData[radiaIndex])

    def getRadiosDataList(self, radiaIndex:int) -> list:
        return self.metaData[radiaIndex]