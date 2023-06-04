#CARREGA AS DEPENDENCIAS
import cv2
import time
import numpy as np
# CORES DAS CLASSES 
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0),(255, 0, 0)]

#CARREGA AS CLASSES
class_names = []
with open("Testes\coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# CAPTURA DE VIDEO
cap = cv2.VideoCapture(0)

# CARREGANDO PESOS DA REDE NEURAL
net = cv2.dnn.readNet("Testes\yolov4.weights", "Testes\yolov4.cfg")


# SETANDO OS PARAMETROS DA REDE NEURAL
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

# LENDO OS FRAMES DO VIDEO
while True:

    #CAPTURA DO VIDEO
    _, frame = cap.read()

    # COMEÇO  DA CONTAGEM DOS MS
    start = time.time()

    # DETECÇÃO
    classes, scores, boxes = model.detect(frame, 0.01, 0.02)
    print(classes)
    # FIM DA CONTAGEM DOS MS
    end = time.time()

    # PERCORRER TODAS AS DETECÇÕES
    for (classid, score, box) in zip(classes, scores, boxes):

        # GERANDO UMA COR PARA A CLASSE 
        color = COLORS[int(classid) % len(COLORS)]

        # PEGANDO O NOME DA CLASSE PELO ID E SEU SCORE DE ACURACIA
        label = f"{class_names[classes[0]]} : {score}"

        # DESENHANDO A BOX DE DETECÇÃO 
        cv2.rectangle(frame, box, color, 2)

        # ESCREVENDO O NOME DA CLASSE EM CIMA DA BOX DO OBJETO
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,2)

    # CALCULANDO O TEMPO QUE LEVOU PARA A DETECÇÃO
    fps_label = f"FPS: {round((1.0/(end - start)),2)}"

    # ESCREVENDO O FPS NA IMAGEM
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # MOSTRANDO IMAGEM
    cv2.imshow("detections", frame)

    # ESPERA DA RESPOSTA
    if cv2.waitKey(1) == 27:
        break

# LIBERAÇÃO DA CAMERA E DETROI AS JANELAS
cap.release()
cv2.destroyAllWindows()


