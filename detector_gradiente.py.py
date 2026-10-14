import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.metrics import structural_similarity as ssim

def convolucao2d(imagem: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Realiza a convolução 2D de uma imagem com um kernel de forma manual (sem scipy/cv2).
    """
    h_img, w_img = imagem.shape
    h_k, w_k = kernel.shape
    pad_h = h_k // 2
    pad_w = w_k // 2

    # Preenchimento de borda refletida para evitar artefatos nos limites
    img_pad = np.pad(imagem, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge').astype(np.float64)
    saida = np.zeros((h_img, w_img), dtype=np.float64)

    # Inversão do kernel conforme a definição formal de convolução
    k_rot = np.flipud(np.fliplr(kernel))

    for i in range(h_img):
        for j in range(w_img):
            regiao = img_pad[i:i + h_k, j:j + w_k]
            saida[i, j] = np.sum(regiao * k_rot)

    return saida

def filtro_gaussiano(imagem: np.ndarray) -> np.ndarray:
    """
    Aplica filtro Gaussiano 3x3 para atenuação de ruído de alta frequência.
    """
    k_gauss = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float64) / 16.0
    return convolucao2d(imagem, k_gauss)

def calcular_gradiente(imagem: np.ndarray, operador: str = 'prewitt'):
    """
    Calcula magnitude e direção do gradiente utilizando os operadores Prewitt ou Scharr.
    """
    if operador == 'prewitt':
        gx_kernel = np.array([[-1, 0, 1],
                              [-1, 0, 1],
                              [-1, 0, 1]], dtype=np.float64)
        gy_kernel = np.array([[-1, -1, -1],
                              [ 0,  0,  0],
                              [ 1,  1,  1]], dtype=np.float64)
    elif operador == 'scharr':
        gx_kernel = np.array([[-3,   0,  3],
                              [-10,  0, 10],
                              [-3,   0,  3]], dtype=np.float64)
        gy_kernel = np.array([[-3, -10, -3],
                              [ 0,   0,  0],
                              [ 3,  10,  3]], dtype=np.float64)
    else:
        raise ValueError("Operador inválido. Escolha 'prewitt' ou 'scharr'.")

    gx = convolucao2d(imagem, gx_kernel)
    gy = convolucao2d(imagem, gy_kernel)

    magnitude = np.sqrt(gx**2 + gy**2)

    # O sinal negativo de gy alinha as coordenadas de matriz ao plano cartesiano (y subindo)
    epsilon = 1e-8
    theta_rad = np.arctan2(-gy, gx + epsilon)
    theta_deg = (np.rad2deg(theta_rad) + 180.0) % 180.0

    return gx, gy, magnitude, theta_deg

def supressao_nao_maximos(magnitude: np.ndarray, angulo: np.ndarray, limiar_corte: float = 0.08) -> np.ndarray:
    """
    Implementação do método 2.b.1: seleção de máximos locais usando os dois vizinhos mais próximos.
    """
    h, w = magnitude.shape
    borda_binaria = np.zeros((h, w), dtype=np.uint8)

    # Limiar mínimo relativo para filtrar variações espúrias em regiões uniformes
    limiar_abs = limiar_corte * np.max(magnitude)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            m_atual = magnitude[i, j]
            if m_atual < limiar_abs:
                continue

            th = angulo[i, j]

            # Mapeamento de vizinhos na vizinhança 3x3:
            # A:[i-1,j-1]  B:[i-1,j]  C:[i-1,j+1]
            # D:[i,  j-1]  E:[i,  j]  F:[i,  j+1]
            # G:[i+1,j-1]  H:[i+1,j]  I:[i+1,j+1]

            if (0.0 <= th < 22.5) or (157.5 <= th <= 180.0):
                # Eixo horizontal: compara com D e F
                v_x = magnitude[i, j - 1]
                v_y = magnitude[i, j + 1]
            elif 22.5 <= th < 67.5:
                # Diagonal 45°: compara com G e C
                v_x = magnitude[i + 1, j - 1]
                v_y = magnitude[i - 1, j + 1]
            elif 67.5 <= th < 112.5:
                # Eixo vertical 90°: compara com H e B
                v_x = magnitude[i + 1, j]
                v_y = magnitude[i - 1, j]
            else:  # 112.5 <= th < 157.5
                # Diagonal 135°: compara com A e I
                v_x = magnitude[i - 1, j - 1]
                v_y = magnitude[i + 1, j + 1]

            # Condição de máximo local
            if m_atual > v_x and m_atual > v_y:
                borda_binaria[i, j] = 1

    return borda_binaria

def processar_imagem(caminho_imagem: str):
    if not os.path.exists(caminho_imagem):
        print(f"Arquivo não encontrado: {caminho_imagem}")
        return

    # Leitura e conversão para escala de cinza
    img_bgr = cv2.imread(caminho_imagem)
    if len(img_bgr.shape) == 3:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_bgr

    img_norm = img_gray.astype(np.float64) / 255.0

    # Etapa 1: Filtragem passa-baixa
    img_filtrada = filtro_gaussiano(img_norm)

    # Etapa 2: Gradiente Prewitt
    _, _, mag_p, ang_p = calcular_gradiente(img_filtrada, operador='prewitt')
    bordas_prewitt = supressao_nao_maximos(mag_p, ang_p, limiar_corte=0.10)

    # Etapa 3: Gradiente Scharr
    _, _, mag_s, ang_s = calcular_gradiente(img_filtrada, operador='scharr')
    bordas_scharr = supressao_nao_maximos(mag_s, ang_s, limiar_corte=0.10)

    # Etapa 4: Canny (Scikit-Image)
    bordas_canny = canny(img_filtrada, sigma=1.0).astype(np.uint8)

    # Etapa 5: Cálculo do SSIM
    ssim_prewitt = ssim(bordas_prewitt.astype(np.float64), bordas_canny.astype(np.float64), data_range=1.0)
    ssim_scharr = ssim(bordas_scharr.astype(np.float64), bordas_canny.astype(np.float64), data_range=1.0)

    nome_base = os.path.splitext(os.path.basename(caminho_imagem))[0]
    print(f"[{nome_base}] SSIM Prewitt vs Canny: {ssim_prewitt:.4f} | SSIM Scharr vs Canny: {ssim_scharr:.4f}")

    # Exibição e salvamento dos resultados
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title(f"Original ({nome_base})")
    axes[0].axis('off')

    axes[1].imshow(bordas_prewitt, cmap='gray')
    axes[1].set_title(f"Prewitt Manual\nSSIM: {ssim_prewitt:.4f}")
    axes[1].axis('off')

    axes[2].imshow(bordas_scharr, cmap='gray')
    axes[2].set_title(f"Scharr Manual\nSSIM: {ssim_scharr:.4f}")
    axes[2].axis('off')

    axes[3].imshow(bordas_canny, cmap='gray')
    axes[3].set_title("skimage Canny")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(f"resultado_{nome_base}.png", dpi=200)
    plt.close()

if __name__ == '__main__':
    imagens = [
        "moedas.png",
        "Lua1_gray.jpg",
        "chessboard_inv.png",
        "img02.jpg"
    ]

    for img in imagens:
        processar_imagem(img)