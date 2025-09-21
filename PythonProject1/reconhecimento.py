import cv2, dlib, numpy as np, pickle, os, time

PREDICTOR = "shape_predictor_5_face_landmarks.dat"
RECOG = "dlib_face_recognition_resnet_model_v1.dat"
DB_FILE = "db.pkl"
THRESH = 0.6

# Carrega banco de dados ou cria vazio
db = pickle.load(open(DB_FILE,"rb")) if os.path.exists(DB_FILE) else {}

# Detector, preditor e reconhecimento facial
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR)
facerec = dlib.face_recognition_model_v1(RECOG)

cap = cv2.VideoCapture(0)

print('|' + '-' * 30 + '|')
print("|" + '    Reconhecimento Facial     ' + "|")
print('|' + '-' * 30 + '|')
print()

print("Pressione:")
print("(r) cadastrar usuário")
print("(v) validar usuário")
print("(q) sair")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb, 1)

    # Exibe número de cadastros na tela
    cv2.putText(frame, f"Cadastros: {len(db)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    for face in faces:
        shape = predictor(rgb, face)
        descriptor = np.array(facerec.compute_face_descriptor(rgb, shape))

        # Nome padrão
        name = "Desconhecido"
        min_dist = 1.0

        for user, db_desc in db.items():
            dist = np.linalg.norm(db_desc - descriptor)
            if dist < min_dist and dist < THRESH:
                min_dist = dist
                name = user

        # Desenha sempre o quadrado + texto
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Mensagem de validação quando em modo validação e reconhecido
        key = cv2.waitKey(1) & 0xFF
        if key == ord('v') and name != "Desconhecido":
            msg = f"O usuário {name} está validado com sucesso!"
            cv2.putText(frame, msg, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Reconhecimento Facial", frame)
            print(msg)
            time.sleep(1)  # mostra mensagem por 1 segundo
            break

    cv2.imshow("Reconhecimento Facial", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Cadastro
        user_name = input("Digite o nome do usuário: ")
        if faces:
            shape = predictor(rgb, faces[0])
            descriptor = np.array(facerec.compute_face_descriptor(rgb, shape))
            db[user_name] = descriptor
            pickle.dump(db, open(DB_FILE,"wb"))
            print(f"Usuário {user_name} cadastrado com sucesso!")
    elif key == ord('v'):
        print("Modo de validação ativo. Coloque o rosto na frente da câmera.")

cap.release()
cv2.destroyAllWindows()
