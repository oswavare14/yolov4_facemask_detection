import numpy as np
import sys
import cv2

#global variables and global enviroment
classes = ['CON MASCARILLA', 'SIN MASCARILLA']
whT = 320

# import del .cfg .weights ya entrenados en el darknet
modelConfiguration = 'data/custom-yolov4-detector.cfg'
modelWeights = 'data/custom-yolov4-detector_best.weights'

# configura los networks
# Esto se usa para cargar los modelos entrenados usando el marco DarkNet. Necesitamos proporcionar dos argumentos aquí también. Una de las rutas a los pesos del modelo y la otra es la ruta al archivo de configuración del modelo 
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
# Configuracion de backend y targeta cpu
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > 0.5:
                w,h = int(det[2]*wT),int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2),int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox,confs,0.5,nms_threshold=0.3)

    for i in indices:
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        if(classIds[i]==0):
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,68),3)
            cv2.putText(img,f'{classes[classIds[i]]} {int(confs[i]*100)}%',
                    (x,y-10),cv2.FONT_ITALIC,0.6,(0,255,68),2)
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(img,f'{classes[classIds[i]]} {int(confs[i]*100)}%',
                        (x,y-10),cv2.FONT_ITALIC,0.6,(0,0,255),2)

def main(cap,flag):
    if flag == 0:
        while True:
            success, img = cap.read()
            #blob prepara la imagen en el formato correcto para introducirla en el modelo 
            blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)

            net.setInput(blob)

            layerNames = net.getLayerNames()
            outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
            # forwward sirve para propagar hacia adelante el blob a través del modelo, lo que nos da todas las salidas.
            outputs = net.forward(outputNames)
            findObjects(outputs, img)
            cv2.imshow('Prediction', img)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
    else:
        blob = cv2.dnn.blobFromImage(cap, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)

        layerNames = net.getLayerNames()
        outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

        outputs = net.forward(outputNames)
        findObjects(outputs, cap)

        cv2.imshow('Prediction', cap)
        cv2.waitKey(0)

def commands():
    flag = 0
    if sys.argv[1] == '--help':
        print("$ pip install opencv-python             #install opencv dependencies")
        print("$ python detect.py --help               #list all commands")
        print("$ python detect.py --video video.mp4    #analyze a video for object-detection")
        print("$ python detect.py --image image.png    #analyze a image for object-detection")
        print("$ python detect.py --webcam             #analyze webcam for object-detection \n\n"
              "IMPORTANT!!!")
        print("while running the program on video or webcam press the key {q} to stop the program")
        exit()
    elif sys.argv[1] == '--image':
        try:
            if sys.argv[2] != None:
                cap = cv2.imread(sys.argv[2])
                flag = 1
        except Exception:
            print("Tiene que brindar el path de la imagen")
            exit()
    elif sys.argv[1] == '--video':
        try:
            if sys.argv[2] != None:
                cap = cv2.VideoCapture(sys.argv[2])

        except Exception:
            print("Tiene que brindar el path del video")
            exit()
    elif sys.argv[1] == '--webcam':
        cap = cv2.VideoCapture(0)

    main(cap,flag)

if __name__ == "__main__":
    commands()
