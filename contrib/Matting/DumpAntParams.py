from pickle import GLOBAL
import numpy as np
from numpy.core.fromnumeric import shape

GLOBAL_paramsmap = {}

def bytes_fill(buf:bytes, size:int):
    if len(buf) > size:
        print("ERROR! size " + str(len(buf)) + " over than " + str(size))
    return buf + b'\0' * (size - len(buf))
# ### 
#     |name(32b)|dim(1b)|shape(8*2b)|data|
#     ...    
#     |name(32b)|dim(1b)|shape(8*2b)|data|
# ###

def Encryption(header):
    encKey = [123,214,243,200,12,31, 1,22,56,89,87,53,16,48,77]
    filebytes = bytearray(header)
    for inx in range(len(filebytes)):
        keyInx = inx % len(encKey)
        filebytes[inx] ^= encKey[keyInx]
    return bytes(filebytes)

def dumpFloat32or16(val, saveAsf16:bool = False):
    header = bytes()
    # cflag 16bit 
    # cflag[0:2]: 0:32f 1:16f, 2:q8, 3:qc8
    if val.dtype == np.float32:
        cflag = 1 if saveAsf16 else 0
        header += cflag.to_bytes(length=2, byteorder='big', signed=False)
    # dim
    dim = len(val.shape)
    header += (dim.to_bytes(length=1, byteorder='big', signed=False))
    # shape
    data_size = 1
    buf = bytes()
    for x in val.shape:
        buf += (x.to_bytes(length=4, byteorder='big', signed=True))
        data_size = data_size * x
    header += bytes_fill(buf, 8 * 4)
    # data
    if saveAsf16:
        valBuf = val.tobytes()
        buf16 = [ 0 ] * int(len(valBuf) / 2)
        for buf_i in range(int(len(valBuf) / 4)):
            srcInx = buf_i * 4
            dstInx = buf_i * 2
            buf16[dstInx] = valBuf[srcInx + 2]
            buf16[dstInx + 1] = valBuf[srcInx + 3]
        header += bytes(buf16)
    else:
        header += val.tobytes()
    return header

def dump_arrmaps(params_map:dict, filename:str, enc:bool = False, saveAsf16:bool = False):
    magicNum = 2106
    header = bytes()
    for key, val in params_map.items():
        # name
        namebuff = key.encode() + "\0".encode()
        header += (len(namebuff).to_bytes(length=1, byteorder='big', signed=False))
        header += namebuff
        if type(val) == dict:
            
            header += dumpFloat32or16(val, saveAsf16)
        else:
            header += dumpFloat32or16(val, saveAsf16)

        
    tensor_count = len(params_map).to_bytes(length=2, byteorder='big', signed=False)
    magicNum = magicNum.to_bytes(length=2, byteorder='big', signed=False)
    header = magicNum + tensor_count + header

    save_file=open(filename, "wb")
    if enc:
        save_file.write(Encryption(header))
    else:
        save_file.write(header)
    
    save_file.close()
