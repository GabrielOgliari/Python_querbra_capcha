import cv2

imagem  = cv2.imread("TesteMetodo/imagem2.png")

# Transformar a imagem em escala de cinza
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

_, imagem_tratada = cv2.threshold(imagemCinza, 127, 255, cv2.THRESH_BINARY_INV or cv2.THRESH_OTSU)
cv2.imwrite('imagem_Tratada2.png', imagem_tratada)