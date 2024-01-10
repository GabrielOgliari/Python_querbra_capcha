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

if __name__ == '__main__': # Se tiver importando o arquivo, não executa o código abaixo se tiver executando o arquivo, executa o código abaixo
    tratarImagem('TesteMetodo')