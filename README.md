# DAA DroneDetection — Simulação e Captura com Cosys-AirSim

Ferramentas Python para simular aproximação de drones intrusos no [Cosys-AirSim](https://github.com/Cosys-Lab/Cosys-AirSim) e capturar dados para treinamento de sistemas de Detecção e Classificação de Aeronaves (DAA).

---

## Estrutura do Projeto

```
DAA_DroneDetection/
├── tools/
│   ├── experiment_controller.py   # Controlador principal de experimentos
│   ├── scenarios.py               # Definições de cenários (26 cenários pré-configurados)
│   └── screen_capture.py          # Captura de tela em tempo real (monitor de execução)
├── config/
│   ├── cosys_airsim_settings.json # Settings para Cosys-AirSim (Documents/AirSim)
│   └── materials.csv               # Lista de materiais para segmentação (Documents/AirSim)
├── requirements.txt
└── README.md
```

---

## Pré-requisitos

| Software | Versão mínima | Observações |
|---|---|---|
| Python | 3.7+ | [python.org](https://www.python.org/downloads/) |
| Git | qualquer | Para clonar o repositório |
| Cosys-AirSim | — | Simulador Unreal — veja seção abaixo |

> [!NOTE]
> Testado em **Windows 10/11** com Python 3.11 e Cosys-AirSim (Unreal 5.x).

---

## 1. Instalar o Cosys-AirSim

1. Siga a documentação oficial: [Cosys-AirSim — How to Get It](https://cosys-lab.github.io/Cosys-AirSim/) (binário pré-compilado ou build from source).
2. Copie os arquivos de configuração para `Documents\AirSim\` (o simulador só lê desse path):

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\Documents\AirSim" | Out-Null
Copy-Item config\cosys_airsim_settings.json "$env:USERPROFILE\Documents\AirSim\settings.json"
Copy-Item config\materials.csv "$env:USERPROFILE\Documents\AirSim\materials.csv"
```

> [!IMPORTANT]
> - **settings.json**: usa `SettingsVersion: 2.0` (exigido pelo Cosys-AirSim). Define o drone **`Drone1`** e câmeras RGB, Depth e Segmentation.
> - **materials.csv**: evita o erro "Material list was not found" na inicialização do stencil (segmentação). Se o ambiente Unreal do Cosys-AirSim vier com outro `materials.csv`, use o dele em vez do nosso.

---

## 2. Clonar o Repositório

```powershell
git clone https://github.com/andrekuros/DAA_DroneDetection.git
cd DAA_DroneDetection
```

---

## 3. Criar e Ativar o Ambiente Virtual

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> [!NOTE]
> Se o PowerShell bloquear a execução de scripts, primeiro execute:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

---

## 4. Instalar Dependências

Instale tudo com pip (o cliente Cosys-AirSim instala numpy e rpc-msgpack automaticamente):

```powershell
pip install -r requirements.txt
```

Ou manualmente:

```powershell
pip install cosysairsim opencv-python mss pywin32 numpy
```

**Alternativa (from source):** clone [Cosys-AirSim](https://github.com/Cosys-Lab/Cosys-AirSim.git) e instale o Python client:

```powershell
cd path\to\Cosys-AirSim\PythonClient
pip install .
```

### Verificar instalação

```powershell
python -c "import cosysairsim; import cv2; import mss; print('Tudo OK!')"
```

---

## 5. Uso

### 5.1 Listar cenários disponíveis

```powershell
# Só lista os 26 cenários (não requer simulador)
python tools\experiment_controller.py --dry-run
```

Saída esperada:

```
 #  Nome                              Pos Inicial NED (x,y,z)       Resumo
────────────────────────────────────────────────────────────────────────────
 1  frontal_clear_day                 (+200.0,   +0.0,   -5.0)  [frontal_clear_day] dist=200m ...
 2  lateral_right_clear_day          ...
...
26  worst_case                        ...
Total: 26 cenários | Observador NED: (0.0, 0.0, -5.0)
```

### 5.2 Executar um cenário único

1. Inicie o ambiente Cosys-AirSim (Unreal com plugin Cosys-AirSim).
2. Aguarde o mapa carregar completamente.
3. Execute:

```powershell
python tools\experiment_controller.py --scenario frontal_clear_day
```

### 5.3 Gerar dataset completo (todos os cenários)

```powershell
python tools\experiment_controller.py --all --output-dir dataset\
```

### 5.4 Opções da linha de comando

| Argumento | Descrição | Padrão |
|---|---|---|
| `--dry-run` | Lista cenários sem conectar ao simulador | — |
| `--list` | Idem a --dry-run | — |
| `--scenario <nome>` | Executa um cenário pelo nome | — |
| `--all` | Executa todos os cenários | — |
| `--output-dir <dir>` | Pasta de saída do dataset | `dataset/` |
| `--observer-pos X Y Z` | Posição NED do observador em metros | `0 0 -5` |
| `--stop-dist <m>` | Distância para encerrar aproximação | `5.0` |
| `--capture-interval <s>` | Intervalo entre frames (ex: 0.1 = 10 fps) | `0.1` |
| `--image-types scene depth seg` | Tipos de imagem a capturar | `scene depth seg` |
| `--vehicle <nome>` | Nome do veículo no settings.json | `Drone1` |
| `--camera <nome>` | Nome da câmera | `front_center` |
| `--ip <endereço>` | IP do host Cosys-AirSim | `127.0.0.1` |

### 5.5 Monitorar visualmente a execução

O `screen_capture.py` captura a janela do simulador em tempo real:

```powershell
# Em um terminal separado, antes de iniciar o experimento:
python tools\screen_capture.py --title "Unreal"
```

Controles da janela de captura: `Q`/`ESC` = sair | `S` = salvar frame | `P` = pausar | `+`/`-` = zoom

---

## 6. Estrutura do Dataset Gerado

```
dataset/
├── run_001_frontal_clear_day/
│   ├── rgb/
│   │   ├── frame_000001.png   ← imagem RGB (Scene)
│   │   ├── frame_000002.png
│   │   └── ...
│   ├── depth/
│   │   ├── frame_000001.pfm   ← profundidade em metros (float32)
│   │   └── ...
│   ├── seg/
│   │   ├── frame_000001.png   ← máscara de segmentação (RGB)
│   │   └── ...
│   └── telemetry.csv          ← posição/velocidade/orientação por frame
├── run_002_lateral_right_clear_day/
│   └── ...
└── ...
```

### Colunas do `telemetry.csv`

| Coluna | Descrição |
|---|---|
| `frame` | Índice do frame |
| `timestamp_s` | Timestamp Unix |
| `x_m, y_m, z_m` | Posição NED em metros |
| `vx_ms, vy_ms, vz_ms` | Velocidade NED em m/s |
| `roll_rad, pitch_rad, yaw_rad` | Orientação em radianos |
| `wx, wy, wz` | Velocidade angular |
| `scenario` | Nome do cenário |

---

## 7. Cenários Pré-configurados

Os 26 cenários cobrem as principais variações de condição para DAA:

| Categoria | Cenários incluídos |
|---|---|
| **Geometria** | Frontal, lateral (D/E), diagonal NE, por trás |
| **Altitude** | Mesmo nível, +20m acima, -10m abaixo, +50m (mergulho) |
| **Distância inicial** | 50m, 100m, 200m, 400m |
| **Velocidade** | 2, 5, 8, 10, 15, 20 m/s |
| **Horário** | Amanhecer (06h), Dia (14h), Entardecer (18h30), Noite (02h) |
| **Clima** | Limpo, chuva leve, chuva forte, neblina, poeira, neve |
| **Vento** | Sem vento, cruzado 10 m/s, de cauda 15 m/s |
| **Combinados** | Chuva noturna, pior caso (noite + chuva + neblina + vento) |

Para adicionar novos cenários, edite `tools/scenarios.py` e adicione à lista `SCENARIOS`.

---

## 8. Sistema de Coordenadas (NED)

O Cosys-AirSim usa **NED (North-East-Down)**:
- **X** = Norte (positivo para frente)
- **Y** = Leste (positivo para direita)
- **Z** = Para baixo (negativo = altitude acima do solo)

Exemplo: posição `(0, 0, -5)` = 5 metros acima do ponto de origem.

O ângulo de **azimute** dos cenários segue a convenção geográfica:
- `0°` = Norte (aproximação frontal)
- `90°` = Leste (aproximação pela direita)
- `180°` = Sul (aproximação por trás)
- `270°` = Oeste (aproximação pela esquerda)

---

## 9. Solução de Problemas

### ❌ Cliente não conecta ao Cosys-AirSim
- Verifique que o ambiente Unreal com Cosys-AirSim está rodando antes do controlador
- Confirme que o `settings.json` está em `Documents\AirSim\`
- Use `--ip 127.0.0.1` (padrão) ou o IP da máquina onde o simulador roda

### ❌ `ModuleNotFoundError: No module named 'cosysairsim'`
- Instale com: `pip install cosysairsim` ou use o PythonClient do repositório Cosys-AirSim

### ❌ `KeyError: vehicle not found`
- O nome do veículo no `settings.json` deve coincidir com `--vehicle` (padrão: `Drone1`)

### ❌ Imagens em branco / pretas
- Aguarde o ambiente Unreal carregar completamente antes de rodar o script
- Verifique `CaptureSettings` no `settings.json` para os ImageTypes usados

### ❌ "You are using newer version of AirSim with older version of settings.json"
- O projeto já usa `SettingsVersion: 2.0` em `config\cosys_airsim_settings.json`. Copie de novo para `Documents\AirSim\settings.json` e reinicie o simulador.

### ❌ "Material list was not found: .../materials.csv"
- Copie `config\materials.csv` para `Documents\AirSim\materials.csv`. Se o seu ambiente Cosys-AirSim/Unreal tiver um `materials.csv` próprio, use esse em vez do nosso.

### ❌ `PSSecurityException` ao ativar o venv
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Dependências

| Pacote | Uso |
|---|---|
| `cosysairsim` | API Python do [Cosys-AirSim](https://github.com/Cosys-Lab/Cosys-AirSim) |
| `numpy` | Manipulação de arrays de imagem |
| `opencv-python` | Exibição e gravação de vídeo (`screen_capture.py`) |
| `mss` | Captura de tela de alta velocidade |
| `pywin32` | Seleção de janela por título (Windows) |
