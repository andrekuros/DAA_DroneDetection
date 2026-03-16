# DAA DroneDetection — Simulação e Captura com Colosseum/AirSim

Ferramentas Python para simular aproximação de drones intrusos no Colosseum (fork do AirSim) e capturar dados para treinamento de sistemas de Detecção e Classificação de Aeronaves (DAA).

---

## Estrutura do Projeto

```
DAA_DroneDetection/
├── tools/
│   ├── experiment_controller.py   # Controlador principal de experimentos
│   ├── scenarios.py               # Definições de cenários (26 cenários pré-configurados)
│   └── screen_capture.py          # Captura de tela em tempo real (monitor de execução)
├── config/
│   └── colosseum_settings.json    # Settings pré-configurado para Colosseum
├── requirements.txt
└── README.md
```

---

## Pré-requisitos

| Software | Versão mínima | Observações |
|---|---|---|
| Python | 3.10+ | [python.org](https://www.python.org/downloads/) |
| Git | qualquer | Para clonar o repositório |
| Colosseum | 2.3.0+ | Simulador — veja seção abaixo |

> [!NOTE]
> Testado em **Windows 10/11** com Python 3.11 e Colosseum 2.3.0.

---

## 1. Instalar o Colosseum

1. Baixe o binário do Colosseum em: https://github.com/CodexLabsLLC/Colosseum/releases
2. Extraia em qualquer pasta (ex: `C:\Colosseum\`)
3. Copie o arquivo de configuração do projeto para `Documents\Colosseum\`:

```powershell
Copy-Item config\colosseum_settings.json "$env:USERPROFILE\Documents\Colosseum\settings.json"
```

> [!IMPORTANT]
> O `settings.json` configura um drone chamado **`Drone1`** com câmeras RGB (640×480), Depth e Segmentation ativas. Sem ele, o controlador não encontra o veículo.

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

### 4.1 Dependências padrão (pip)

```powershell
pip install numpy mss opencv-python pywin32 msgpack-rpc-python "msgpack==0.6.2"
```

> [!IMPORTANT]
> É obrigatório fixar `msgpack==0.6.2`. Versões `>= 1.0` removeram o argumento `encoding` usado internamente pelo `msgpack-rpc-python`, causando `TypeError` ao conectar ao Colosseum.

### 4.2 Módulo `airsim` do Colosseum

> [!IMPORTANT]
> O pacote `airsim` no PyPI está **quebrado para Python 3.10+**. Use o script abaixo para instalar diretamente do repositório do Colosseum.

```powershell
# Cria pasta do módulo no venv
$dest = ".\venv\Lib\site-packages\airsim"
New-Item -ItemType Directory -Force $dest | Out-Null

# Baixa os 5 arquivos do módulo airsim
$base = "https://raw.githubusercontent.com/CodexLabsLLC/Colosseum/main/PythonClient/airsim"
@("__init__.py", "client.py", "types.py", "utils.py", "pfm.py") | ForEach-Object {
    Invoke-WebRequest "$base/$_" -OutFile "$dest\$_" -UseBasicParsing
    Write-Host "OK: $_"
}
```

### 4.3 Verificar instalação

```powershell
python -c "import airsim; import cv2; import mss; print('Tudo OK!')"
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

1. Abra o Colosseum (execute o arquivo `.exe` do binário baixado)
2. Aguarde o mapa carregar completamente
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
| `--ip <endereço>` | IP do host Colosseum | `127.0.0.1` |

### 5.5 Monitorar visualmente a execução

O `screen_capture.py` captura a janela do Colosseum em tempo real para você ver o que está acontecendo:

```powershell
# Em um terminal separado, antes de iniciar o experimento:
python tools\screen_capture.py --title "Colosseum"
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

O AirSim/Colosseum usa **NED (North-East-Down)**:
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

### ❌ `airsim` não conecta ao Colosseum
- Verifique que o Colosseum está rodando antes de executar o controlador
- Confirme que o `settings.json` foi copiado para `Documents\Colosseum\`
- Tente `--ip 127.0.0.1` (padrão) ou o IP da máquina com o simulador

### ❌ `ModuleNotFoundError: No module named 'airsim'`
- Execute o script de instalação manual da seção **4.2**

### ❌ `KeyError: vehicle not found`
- O nome do veículo no `settings.json` deve bater com `--vehicle` (padrão: `Drone1`)

### ❌ Imagens em branco / pretas
- Aguarde o Colosseum carregar completamente antes de rodar o script
- Verifique se o `ImageType` está configurado no `settings.json` (`CaptureSettings`)

### ❌ `PSSecurityException` ao ativar o venv
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Dependências

| Pacote | Uso |
|---|---|
| `airsim` | API de controle do Colosseum |
| `numpy` | Manipulação de arrays de imagem |
| `opencv-python` | Exibição e gravação de vídeo (`screen_capture.py`) |
| `mss` | Captura de tela de alta velocidade |
| `pywin32` | Seleção de janela por título (Windows) |
| `msgpack-rpc-python` | Transporte RPC do protocolo AirSim |
