# \# 🔬 PCB Inspector — Sistema Inteligente de Inspeção de Qualidade e Triagem Automática em Placas de Circuito Impresso

# 

# > Solução autoral de Visão Computacional aplicada à Indústria 4.0 para detecção automatizada de falhas e análise estatística de conformidade em linhas de montagem de PCBs.

# 

# \---

# 

# \## 👥 Integrantes da Equipe

# \* \*\*\[João Pedro Bezerra]\*\* 

# \* \*\*\[Adan Cristyan]\*\* 

# \* \*\*\[Bryan Rongelli]\*\* 

# \* \*\*\[Tatiane Oliveira]\*\* 
# 

# \---

# 

# \## 📌 Escopo do Projeto e Definição do Modelo

# 

# \### Tema Escolhido e Tarefa YOLO

# \* \*\*Tema:\*\* Tema 3 — Inspeção de Qualidade / Triagem Automática em Esteira (Disciplina de Processamento Digital de Imagens).

# \* \*\*Tarefa do YOLO:\*\* Detecção de Objetos (Object Detection — `detect`).

# 

# \### Modelo Base Utilizado e Justificativa

# \* \*\*Modelo Base:\*\* Arquitetura `YOLOv8` (Variante Nano — `yolov8n.pt`).

# \* \*\*Justificativa:\*\* Em ambientes industriais de triagem em esteira de alta velocidade, a latência de processamento por quadro (throughput) é uma métrica crítica. A variante \*\*Nano\*\* foi selecionada por oferecer o equilíbrio ideal entre precisão de localização espacial e tempo de inferência extremamente baixo (frequentemente < 50ms por imagem), permitindo a execução eficiente e leve mesmo em computadores convencionais sem placas de vídeo dedicadas (GPU).

# 

# \---

# 

# \## 🛠️ Tecnologias Utilizadas

# 

# \* \*\*Back-end:\*\* Python 3.11 / 3.12, FastAPI, Uvicorn (Servidor ASGI)

# \* \*\*Inteligência Artificial:\*\* Ultralytics YOLOv8, OpenCV (Processamento e Manipulação de Imagens), NumPy

# \* \*\*Front-end:\*\* HTML5, CSS3 (Variáveis nativas, Layout Flexbox/Grid), JavaScript (ES6+ Assíncrono), Chart.js (Métricas gráficas dinâmicas)

# 

# \---

# 

# \## 📐 Arquitetura da Aplicação

# 

# A aplicação adota o modelo de arquitetura cliente-servidor totalmente desacoplada através de uma API RESTful. Isso garante que o processamento pesado de inferência executado pela rede neural ocorra em segundo plano e não bloqueie a renderização ou a fluidez da interface do usuário.

# 

# ```mermaid

# graph LR

# &#x20;   subgraph Cliente (Front-end Dashboard)

# &#x20;       A\[Interface Web HTML5/CSS3] <-->|Atualiza KPIs e Gráficos| B\[JavaScript Engine]

# &#x20;   end

# 

# &#x20;   subgraph Servidor (Back-end API)

# &#x20;       B <-->|POST /inspect - Envia Imagem| C\[FastAPI Server]

# &#x20;       C <-->|Processa Matriz OpenCV| D\[YOLOv8 Engine]

# &#x20;       D <-->|Pesos de Inferência| E\[(best.pt)]

# &#x20;   end

# ```

# 

# \---

# 

# \## 📋 Classes Identificadas pelo Modelo (Dataset)

# 

# O modelo foi treinado para identificar com precisão as \*\*7 falhas mais recorrentes\*\* em linhas de soldagem industrial de componentes eletrônicos:

# 

# | ID | Classe no Código | Descrição Visual do Defeito | Severidade Industrial |

# |:--:|:-----------------|:----------------------------|:----------------------|

# | 0  | `damaged`        | Componente físico trincado, quebrado ou deformado. | Crítico |

# | 1  | `lack\_of\_part`   | Ausência total de um componente obrigatório na placa. | Crítico |

# | 2  | `miss\_welding`   | Ponto onde a solda falhou ou não fixou o terminal. | Crítico |

# | 3  | `redundant`      | Componente extra ou resíduo posicionado incorretamente. | Moderado |

# | 4  | `Short\_circuit`  | Filete de solda unindo trilhas que deveriam estar isoladas. | Crítico |

# | 5  | `slug`           | Resíduo esférico ou gotícula de solda solta na placa. | Leve |

# | 6  | `spillover`      | Excesso ou derramamento de solda para além do pad. | Moderado |

# 

# \---

# 

# \## 🧠 Regras de Negócio e Tomada de Decisão

# 

# A API não apenas localiza os objetos, mas aplica regras rígidas para tomada de decisão em tempo real (Triagem Automática):

# 

# 1\. \*\*Nota de Corte (Confidence Threshold):\*\* Definida em `0.25`. Detecções com grau de certeza inferior a 25% são descartadas automaticamente pelo algoritmo para evitar falsos positivos.

# 2\. \*\*Critério de Reprovação Semafórica:\*\*

# &#x20;  \* \*\*✅ Aprovada (Verde):\*\* Zero anomalias detectadas. A placa está em conformidade absoluta.

# &#x20;  \* \*\*❌ Reprovada (Vermelho):\*\* Presença de \*\*qualquer\*\* defeito categorizado pelo modelo. O sistema aciona um status de rejeição imediata no painel informando a severidade e os tratamentos recomendados.

# 

# \---

# 

# \## 🔌 Endpoints da API (Documentação Técnico-Científica)

# 

# O backend expõe uma arquitetura de rotas limpa, performática e totalmente assíncrona:

# 

# \* `GET /` : Verifica a conectividade e status inicial da API.

# \* `GET /health` : Retorna a integridade do sistema e confirma se o arquivo de pesos `best.pt` está devidamente carregado em memória.

# \* `POST /inspect` : O endpoint principal. Recebe o upload do arquivo de imagem enviado pelo front-end, transforma o payload em uma matriz de imagem via OpenCV, realiza a inferência de IA via YOLOv8, desenha as caixas delimitadoras (\*bounding boxes\*), codifica o resultado em string Base64 e retorna um JSON contendo a contagem de falhas, tempo de inferência e status da peça.

# \* `GET /stats` : Consolida as estatísticas globais acumuladas da sessão de trabalho atual (Taxa de rejeição percentual, total inspecionado, gráfico de falhas comuns).

# \* `GET /history` : Retorna o histórico cronológico detalhado das últimas 20 inspeções realizadas na esteira.

# \* `DELETE /reset` : Reinicializa todos os contadores estatísticos da sessão industrial para zero.

# 

# \---

# 

# \## 🚀 Como Executar o Projeto do Zero (Guia de Implantação)

# 

# \### Pré-requisitos

# \* Python 3.11 ou 3.12 instalado no Windows.

# \* \*\*Atenção no Instalador do Python:\*\* Certifique-se de marcar a caixa \*\*"Add python.exe to PATH"\*\* na primeira tela do instalador.

# 

# \### 1. Preparação do Ambiente e Direcionamento

# Abra o \*\*PowerShell\*\* ou o \*\*Prompt de Comando\*\*, navegue até a pasta onde descompactou o projeto e entre no diretório do servidor:

# 

# ```bash

# cd "C:\\Users\\nyckv\\Desktop\\Inspetor PCB\\backend"

# ```

# 

# Crie o ambiente virtual isolado para o projeto:

# ```bash

# python -m venv venv

# ```

# 

# \### 2. Ativação do Ambiente e Liberação de Diretivas (PowerShell)

# Se estiver utilizando o \*\*PowerShell\*\*, por segurança o Windows bloqueia a execução de scripts. Libere a permissão temporária para esta janela e ative o ambiente virtual executando:

# 

# ```powershell

# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# .\\venv\\Scripts\\Activate.ps1

# ```

# \*(Você confirmará a ativação quando o prefixo `(venv)` aparecer destacado de forma verde no início da sua linha de comando).\*

# 

# \### 3. Instalação Segura das Dependências (Solução de Compilação)

# Para evitar falhas de compilação C++ causadas pelo travamento de versões legadas de pacotes em máquinas limpas, execute a instalação direta das distribuições binárias atualizadas através do comando unificado:

# 

# ```powershell

# pip install fastapi uvicorn ultralytics opencv-python python-multipart numpy

# ```

# 

# \### 4. Execução do Servidor de IA (Back-end)

# Certifique-se de que o arquivo de pesos treinado (`best.pt`) está localizado dentro da pasta `backend`. Em seguida, inicialize a API:

# 

# ```powershell

# python main.py

# ```

# O terminal carregará a rede neural YOLO e confirmará que o servidor HTTP está ativo em `http://127.0.0.1:8000`. \*\*Mantenha esta janela do terminal aberta para processar as imagens.\*\*

# 

# \### 5. Inicialização da Interface Visual (Front-end)

# 1\. Abra o Explorador de Arquivos do Windows e navegue até a pasta `frontend` do projeto (`C:\\Users\\nyckv\\Desktop\\Inspetor PCB\\frontend`).

# 2\. Dê um \*\*duplo clique\*\* sobre o arquivo \*\*`index.html`\*\* (ele será aberto diretamente no seu navegador padrão, como Chrome ou Edge).

# 3\. A interface SPA se conectará de forma invisível à API local. A partir daí, basta arrastar imagens de teste de placas de circuito impresso para a área de upload e analisar em tempo real os mapeamentos de defeitos, os KPIs industriais e os gráficos de controle estatístico!

<img width="2720" height="1680" alt="arquitetura_pcb_inspector" src="https://github.com/user-attachments/assets/90e4a328-88e8-4e1f-bb58-49da97647c90" />

