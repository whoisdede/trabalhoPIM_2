Aqui está uma sugestão de README.md para acompanhar a versão inicial (v0) do seu trabalho. Você pode copiar este texto e salvar em um arquivo chamado README.md na mesma pasta do seu projeto.PIM - Detector de Bordas por Operador Gradiente (v0)Este repositório contém a versão inicial (v0) do Trabalho Prático da disciplina de PIM (Processamento de Imagens). O objetivo principal é a implementação computacional do operador gradiente na detecção de bordas, englobando a construção manual de convoluções matemáticas e a aplicação dos operadores diferenciais de Prewitt e Scharr.📌 Escopo desta Versão (v0)Esta versão atende a todos os requisitos fundamentais estipulados no roteiro da prática:Conversão de Imagem: Transforma imagens coloridas de entrada para tons de cinza.Pré-filtragem: Aplicação de um filtro passa-baixas (Gaussiano 3x3) construído manualmente para atenuar ruídos de alta frequência antes da derivação.Convolução 2D Manual: Implementação do cálculo matemático da convolução sem a utilização de bibliotecas prontas para este fim (como scipy.signal ou cv2.filter2D).Cálculo do Gradiente: Extração das componentes horizontal ($Gx$) e vertical ($Gy$), Magnitude ($M$) e Direção/Ângulo ($D$) utilizando as máscaras de Prewitt e Scharr.Supressão de Não-Máximos Locais: Implementação da regra dos dois vizinhos colineares adjacentes (Item 2.b.1 do complemento) separados em 4 setores (horizontal, vertical e diagonais) para afinar as bordas detectadas.Comparação de Desempenho: Utilização da métrica SSIM (Structural Similarity Index Measure) para comparar a detecção manual com a função skimage.feature.canny.🛠️ Requisitos e Configuração do AmbienteFoi utilizado o Python 3.x para este projeto. Siga os passos abaixo para preparar o ambiente virtual e instalar as dependências necessárias.No Windows (PowerShell)Execute um comando por vez:PowerShell# 1. Crie o ambiente virtual
python -m venv venv

# 2. Libere a execução de scripts no terminal (necessário apenas na primeira vez)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. Ative o ambiente virtual
.\venv\Scripts\Activate.ps1

# 4. Atualize o instalador de pacotes
python -m pip install --upgrade pip

# 5. Instale as bibliotecas necessárias
python -m pip install numpy opencv-python scikit-image matplotlib
No Linux / macOS (Terminal)Bashpython3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy opencv-python scikit-image matplotlib
🚀 Como ExecutarCom o ambiente virtual ativado, coloque as imagens de teste (moedas.png, Lua1_gray.jpg, chessboard_inv.png e img2.jpg) no mesmo diretório do arquivo .py principal e execute:Bashpython detector_gradiente.py
📊 Saída EsperadaA execução do código não possui interface gráfica interativa, mas gera dois tipos de saídas para compor o relatório do trabalho:Console (Terminal):O script exibirá os valores do SSIM calculados entre a implementação manual (Prewitt/Scharr) e o detector Canny de referência, confirmando o índice de similaridade estrutural. Exemplo:Plaintext[moedas] SSIM Prewitt vs Canny: 0.7845 | SSIM Scharr vs Canny: 0.7912
Arquivos de Imagem:O script criará arquivos .png (resultado_moedas.png, resultado_img2.png, etc.) na pasta principal. Cada arquivo é uma prancha contendo a comparação lado a lado:Original (Tons de Cinza)Resultado com Prewitt ManualResultado com Scharr ManualResultado Canny (skimage)📝 Observações sobre os ResultadosConforme discutido na teoria, enquanto o Canny (skimage) utiliza a técnica avançada de limiarização por histerese (conectando bordas fracas e fortes), a nossa implementação (v0) foca na supressão simples de não-máximos locais. Por conta disso, é esperado e academicamente correto que as bordas da nossa implementação apresentem leve descontinuidade em comparação ao Canny, servindo de objeto de análise para o relatório final.
