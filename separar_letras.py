import cv2
import os
import glob

def tratarImagem(pasta_origem='BaseImagens/B_imagens', pasta_destino='BaseImagens/imagensTratadas'):
    arquivos =glob.glob(f'{pasta_origem}/*')
    for arquivo in arquivos:
        imagem  = cv2.imread(arquivo)
        # Transformar a imagem em escala de cinza
        imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
        
        nomeImagem = os.path.basename(arquivo)
        _, imagem_tratada = cv2.threshold(imagemCinza, 127, 255, cv2.THRESH_BINARY_INV or cv2.THRESH_OTSU)
        cv2.imwrite(f'{pasta_destino}/{nomeImagem}', imagem_tratada)

tratarImagem('TesteMetodo')
arquivos = glob.glob('BaseImagens/imagensTratadas/*')
# arquivos = glob.glob('TesteMetodo/*')
for arquivo in arquivos:
    imagem = cv2.imread(arquivo)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    # em preto e branco
    _, nova_imagem = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY_INV)


    # encontrar os contornos de cada letra
    contornos, _ = cv2.findContours(nova_imagem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regiao_letras = []

    # filtrar os contornos que são realmente de letras
    for contorno in contornos:
        (x, y, largura, altura) = cv2.boundingRect(contorno)
        area = cv2.contourArea(contorno)
        if area > 30:
            regiao_letras.append((x, y, largura, altura))
    # if len(regiao_letras) != 4:
    #     continue
    
    #desenhar os contornos e separar as letras em arquivos individuais

    imagem_final = cv2.merge([nova_imagem] * 3)

    i = 0
    for retangulo in regiao_letras:
        x, y, largura, altura = retangulo
        imagem_letra = imagem_final[y-2:y+altura+2, x-2:x+largura+2]
        i += 1
        nome_arquivo = os.path.basename(arquivo).replace(".png", f"letra{i}.png")
        cv2.imwrite(f'BaseImagens/letras/{nome_arquivo}', imagem_letra)
        cv2.rectangle(imagem_final, (x-2, y-2), (x+largura+2, y+altura+2), (0, 255, 0), 1)
    nome_arquivo = os.path.basename(arquivo)
    cv2.imwrite(f"BaseImagens/identificado/{nome_arquivo}", imagem_final)