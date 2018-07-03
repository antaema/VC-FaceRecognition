import cv2
import os
import numpy as np

class FaceDetector:
    # Detecta a face 
    def detect_face(self, img, scaleFactor):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else: 
            gray = img
        face_cascade = cv2.CascadeClassifier('opencv-files/haarcascades/haarcascade_frontalface_alt.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=4)

        if (len(faces) == 0):
            return None, None

        lfaces = []

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            data = (roi_gray,(x,y,w,h))

            # Posicao 0: face - Posicao 1: local da face
            lfaces.append(data)
        return lfaces

class FaceRecognitor:

    def __init__(self, configPath):
        print("[*] Carregando configuracoes ...")
        self.subjects = []
        self.detector = FaceDetector()
        self.subjects.append('')
        self.face_recognizerE = cv2.face.EigenFaceRecognizer_create()
        self.face_recognizerF = cv2.face.FisherFaceRecognizer_create()
        self.face_recognizerL = cv2.face.LBPHFaceRecognizer_create()

        for line in open(configPath).readlines():
            line = line.rstrip()
            self.subjects.append(line)
                       
        print("[*] Configuracoes carregadas.")
        
        print("[*] Preparando dados ...")
        self.faces, self.labels, self.efaces = self.prepare_training_data("training-data")
        self.tfaces, self.tefaces = self.prepare_test_data("test-data")
        print("[*] Dados preparados.")

    def draw_rectangle(self, img, rect):
        (x, y, w, h) = rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    def draw_text(self, img, text, x, y):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    def show_faces(self,img,scaleFactor):
        lfaces = self.detector.detect_face(img, scaleFactor)
        for data in lfaces:
            face, rect = data
            self.draw_rectangle(img,rect) 

    def prepare_test_data(self, data_folder_path):
        dirs = os.listdir(data_folder_path)

        faces = []
        efaces = []
       
        for dir_name in dirs:
            if not dir_name.startswith("s"):
                continue
            label = int(dir_name.replace("s", ""))
            subject_dir_path = data_folder_path + "/" + dir_name
            subject_images_names = os.listdir(subject_dir_path)
            
            for image_name in subject_images_names:
                if image_name.startswith("."):
                    continue
                image_path = "./"+subject_dir_path + "/" + image_name
                image = cv2.imread(image_path, 0)
                cv2.imshow('Identificando faces na imagem ...', image)
                cv2.waitKey(100)
                datas = self.detector.detect_face(image,1.02)
                for data in datas:
                    face, rect = data
                    if face is not None:
                        faces.append(face)
                        efaces.append(cv2.resize(face,(150,150)))
                    
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        return faces, efaces      

    def prepare_training_data(self, data_folder_path):
        dirs = os.listdir(data_folder_path)

        faces = []
        labels = []
        efaces = []
        
        # abre cada pasta e abri todas as imagens
        for dir_name in dirs:
            # toda pasta de pessoa comeca com s
            if not dir_name.startswith("s"):
                continue
            
            # pega a label da pessoa
            label = int(dir_name.replace("s", ""))
            
            # caminho para a pasta contendo as imagems
            subject_dir_path = data_folder_path + "/" + dir_name
            
            # pega o nome das imagems
            subject_images_names = os.listdir(subject_dir_path)
            sumFace = [[],[]]
            # abre cada imagem detecta a face e adiciona a uma lista
            for image_name in subject_images_names:
                
                # ignora arquivos do sistema
                if image_name.startswith("."):
                    continue
                
                # gera caminho da imagem
                image_path = "./"+subject_dir_path + "/" + image_name

                #le a imagem
                image = cv2.imread(image_path, 0)

                #mostra a imagem sendo analisada 
                cv2.imshow('Treinando na imagem ...', image)
                cv2.waitKey(100)
                
                #detecta faces
                datas = self.detector.detect_face(image,1.02)
                for data in datas:
                    face, rect = data
                    sumFace[0].append(face.shape[0])
                    sumFace[1].append(face.shape[1])
                    if face is not None:
                        faces.append(face)
                        efaces.append(cv2.resize(face,(150,150)))
                        labels.append(label)
            
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        print("[*] Mean faces: ", np.mean(np.asarray(sumFace[0])), np.mean(np.asarray(sumFace[1])))
        return faces, labels, efaces      

    def inicializeRecognizers(self):
        self.face_recognizerL.train(self.faces, np.array(self.labels))
        self.face_recognizerE.train(self.efaces, np.array(self.labels))
        self.face_recognizerF.train(self.efaces, np.array(self.labels))

    def predictLBPH(self, test_img):
        
        #make a copy of the image as we don't want to chang original image
        img = test_img.copy()
        #detect face from the image
        datas = self.detector.detect_face(img, 1.02)
        
        for data in datas:    
            face, rect = data
            #predict the image using our face recognizer 
            label, confidence = self.face_recognizerL.predict(face)
            #get name of respective label returned by face recognizer
            label_text = self.subjects[label]
            
            #draw a rectangle around face detected
            self.draw_rectangle(img, rect)
            #draw name of predicted person
            self.draw_text(img, label_text, rect[0], rect[1]-5)
        return img, confidence, label_text

    # Não foi utilizado pois o algoritimo espera todas as imagens do mesmo tamanho
    def predictEigen(self, test_img):
        img = test_img.copy()
        datas = self.detector.detect_face(img, 1.02)

        for data in datas:    
            face, rect = data
            # exige que as entradas tenham mesmo tamanho
            face = cv2.resize(face,(150,150))
            label, confidence = self.face_recognizerE.predict(face)
            label_text = self.subjects[label]
            
            self.draw_rectangle(img, rect)
            self.draw_text(img, label_text, rect[0], rect[1]-5)
        return img, confidence, label_text

    def predictFisher(self, test_img):
        img = test_img.copy()
        datas = self.detector.detect_face(img, 1.02)
        
        for data in datas:    
            face, rect = data
            # exige que as entradas tenham mesmo tamanho
            face = cv2.resize(face,(150,150))
            label, confidence = self.face_recognizerF.predict(face)
            label_text = self.subjects[label]
            
            self.draw_rectangle(img, rect)
            self.draw_text(img, label_text, rect[0], rect[1]-5)
        return img, confidence, label_text

    def test(self):
        self.inicializeRecognizers()
        print("[*] Testando")

        file = open('result.txt', 'w')
        file2 = open('result2.txt', 'w')
        file3 = open('result3.txt', 'w')

        file.write("Image   Method   Validation  Confidence Person \n")
        file2.write("Image   Method   Validation  Confidence Person \n")
        file3.write("Image   Method   Validation  Confidence Person \n")

        for i in range(1,16):
            j = i - 1
            expected = self.subjects[i]
            result1 = self.predictEigen(self.tefaces[j])
            result2 = self.predictFisher(self.tefaces[j])
            result3 = self.predictLBPH(self.tfaces[j])

            validate = []
            if result1[2] == expected:
                validate.append('True')
            else:
                validate.append('False')
            
            if result2[2] == expected:
                validate.append('True')
            else:
                validate.append('False')
            
            if result3[2] == expected:
                validate.append('True')
            else:
                validate.append('False')
        
            file.write(str(i)+"\tEigen\t"+validate[0]+"\t"+ str(result1[1]) + "\t" + result1[2] + "\n")
            file2.write(str(i)+"\tFisher\t"+validate[1]+"\t"+ str(result2[1]) + "\t" + result2[2] + "\n")
            file3.write(str(i)+"\tLBPH\t"+validate[2]+"\t"+ str(result3[1]) + "\t" + result3[2] + "\n")

        file.close() 
        file2.close() 
        file3.close()    
        print("[*] Teste Concluido")

fr = FaceRecognitor('./config')
fr.test()