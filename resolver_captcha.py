from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import cv2
import pickle

from tratar_captcha import tratarImagem   

def quebrar_captcha():
    # importar o modelo treinado e importar o tradutor
    with open('rotulos_modelo.dat', 'rb') as arquivo_tradutor:
        lb = pickle.load(arquivo_tradutor)

    modelo = load_model('modelo_treinado.keras')

    # usar o modelo para resolver o captcha
    tratarImagem("TesteMetodo", "resolver_captcha")

    # ler todas os arquivos da pasta "resolver_captcha"
    arquivos = list(paths.list_images('resolver_captcha'))
    for arquivo in arquivos:
        imagem = cv2.imread(arquivo)
        _, imagem = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY_INV)

        # em preto e branco
        nova_imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)


        # encontrar os contornos de cada letra
        contornos, _ = cv2.findContours(nova_imagem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regiao_letras = []

        # filtrar os contornos que são realmente de letras
        for contorno in contornos:
            (x, y, largura, altura) = cv2.boundingRect(contorno)
            area = cv2.contourArea(contorno)
            if area > 30:
                regiao_letras.append((x, y, largura, altura))
        
        regiao_letras = sorted(regiao_letras, key=lambda x: x[0])
        #desenhar os contornos e separar as letras em arquivos individuais

        imagem_final = cv2.merge([nova_imagem] * 3)
        previsao = []

        i = 0
        # No changes needed
        for retangulo in regiao_letras:
            x, y, largura, altura = retangulo
            imagem_letra = imagem_final[y-2:y+altura+2, x-2:x+largura+2]
            i += 1
            
            # dar a letra para o modelo prever
            imagem_letra = cv2.cvtColor(imagem_letra, cv2.COLOR_BGR2GRAY)
            imagem_letra = resize_to_fit(imagem_letra, 20, 20)

            # adicionar duas dimensão para o Keras poder ler a imagem
            imagem_letra = np.expand_dims(imagem_letra, axis=2)
            imagem_letra = np.expand_dims(imagem_letra, axis=0)

            letra_prevista = modelo.predict(imagem_letra)
            letra_prevista = lb.inverse_transform(letra_prevista)[0]
            previsao.append(letra_prevista)

            

        
        texto_previsao = ''.join(previsao)
        print(texto_previsao)
        #return texto_previsao


if __name__ == '__main__':
    quebrar_captcha()