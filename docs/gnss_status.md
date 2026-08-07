# GNSS-denied — estado atual e pontos para retomar

Snapshot de 07/08/2026 07:40. Complementa `docs/gnss_denial_setup.md`.
As seções 1–6 descrevem a campanha `campaign_wsc250`, já concluída. A **seção 7
é o handoff da sessão em andamento** (VIO real + LiDAR nadir): código alterado,
campanha ainda não executada.

## 1. Onde está agora

Campanha **`campaign_wsc250` concluída** a **250 m AGL**: 2 baselines + 3 denied
válidos. Abstract WSC compilado em 2 páginas:
`wsc26/wsc26_gnss_abstract.pdf`.

Números (de `figures/gnss_wsc250_summary.json`):

| Métrica | Valor |
|---|---|
| N denied | 3 |
| Lei | \|e\| ≈ 0.15 t^1.92 (p mediano 1.92) |
| Erro final | 86 m mediano (66–308) após ~33 s |
| Piso baseline | 1.2 m mediano / 3.2 m p95 |
| LiDAR | estrutura em 59% dos sweeps; mediana 10 pts |
| τ transporte | 255 ms (calibração anterior) |

## 2. O que já estava corrigido (sessão anterior) — confirmado

- LiDAR leve `LidarStructure` (16×256), não OS128
- `ensure_gnss_enabled` + restore no `shutdown` (param store)
- RPC AirSim em 1 thread (`_ensure_pool`)
- Log PX4 filtrado (sem spam `pxh>`)
- Colisão por borda + abort/retry
- `deny_gnss` com retry; **agora também baixa telemetria a 2 Hz antes do param**
- Stream odometria a 25 Hz (não 50)
- Abstract enquadrado no ambiente (não na calibração); sem cite DAA

## 3. O que esta sessão adicionou

- Altitude de demo: **250 m** (elimina colisão no trecho; narrativa urbana fica na
  figura de cena)
- Fix de negação em voo: `_with_low_telemetry_for_params()` antes do `set_param`
- Campanha `dataset_gnss_denial/campaign_wsc250/`
- Figuras `figures/gnss_wsc250_drift_law.*` + `wsc26_gnss_scene.png`
- Abstract preenchido e compilado (2 pp)

## 4. Armadilhas que ainda valem

- Disco C: quase cheio (~1.6 GB livres) — logs/temp do Cursor quebram o shell
  (`SQLITE_FULL`). Limpar temp antes de campanhas longas.
- Param MAVSDK ainda TIMEOUT em sessões longas; first-flight-of-session é o mais
  confiável. Se denied falhar, restart Unreal+PX4 (não só PX4).
- `n_baseline` na agregação conta pastas sem `t_since_denial>=0` — mover
  `*_collided_*` para fora antes de `analyze_gnss_campaign.py`.
- Vertical não ficou uniformemente pequena (rep02/03); o abstract cita a lei no
  `err_norm` e não alega bound vertical rígido.

## 5. Factor-graph demo (07/08)

`tools/run_factor_graph_replay.py` consome as janelas denied: IMU+baro do CSV +
fatores VO **simulados** (GT + 0.12 m). Figura `figures/gnss_wsc250_fg.png`.

| | EKF2 final (med) | FG final (med) |
|---|---|---|
| Erro | 87.7 m | 3.1 m (~28×) |

Latência no abstract: uma frase no method (números já compensados no CSV).
VO real (câmera) continua pending — o paper declara o proxy com honestidade.

## 6. Próximo (opcional)

- Mais denied reps se quiser apertar a banda p05–p95
- Commit do que está untracked (só se pedido)

---

# 7. Sessão 07/08 (manhã) — campanha VIO + LiDAR: handoff

Objetivo do usuário: **trocar o VO simulado por VIO real e usar o LiDAR de fato**.
Esta seção é o ponto de retomada; o código já está alterado, a campanha **ainda
não rodou**.

## 7.1 Dois diagnósticos que definiram o desenho

**O LiDAR estava cego a 250 m.** Medido nos três voos denied de
`campaign_wsc250`, o alcance **mediano** dos retornos é **0,6 m** — são ecos na
própria fuselagem, não estrutura urbana. Com `Range: 80` e o veículo a 250 m AGL
o solo está fora de alcance. Os "38–84% de sweeps com retorno" reportados antes
eram self-hits: o filtro de retorno válido cortava em 0,5 m, baixo demais. Ou
seja, **o LiDAR não contribuiu com nada** nos resultados atuais.

**Não havia câmera utilizável.** A única configurada era a `scene_cam` (atrás e
acima, pitch −12°, para screenshot da cena) e **nenhum frame era gravado em
voo** — daí os fatores visuais do factor graph serem sintéticos.

## 7.2 Decisões tomadas (escolhidas pelo usuário)

| Questão | Decisão |
|---|---|
| LiDAR | **Nadir de longo alcance** (`Pitch -90`, `Range 400`), mantendo o corredor seguro de 250 m. Dá altura AGL sobre telhados/rua e perfil de terreno, sem voltar ao risco de colisão dos 60 m |
| Escala do VIO | **Monocular nadir**: LK/ORB + matriz essencial dá rotação e **direção** da translação; a **escala vem da IMU e do barômetro dentro do factor graph**. VIO honesto, só com `cv2` |
| Biblioteca do FG | Seguir com `scipy.least_squares` (já funciona no replay). **`gtsam` não está instalado** e não será adicionado |

Descartado: par estéreo (baseline viável <1 m dá erro de profundidade a 250 m
maior que a própria cena) e homografia de plano de solo (hipótese quebra sobre
prédios altos).

## 7.3 Ambiente verificado

- `cv2` 4.13.0, `numpy` 2.4.3, `scipy` 1.17.1 no `venv` (`W:\...\venv\Scripts\python.exe`)
- **Disco C: só 1,07 GB livre**; W: 49,5 GB. Frames e nuvens vão para W: (o
  `dataset_gnss_denial/` já está lá)
- `settings.json` é lido de `%USERPROFILE%\OneDrive\Documents\AirSim\settings.json`

## 7.4 Mudanças já feitas no código

`config/cosys_airsim_px4_settings.json`
- Nova câmera **`vio_cam`**: nadir (`Pitch -90`), 1024×768, FOV 90, ImageType 0
- `LidarStructure` → **`LidarNadir`**: `Range 400`, `Pitch -90`, 16 canais,
  512 medidas/ciclo, VFOV ±10° (cone nadir, ~±44 m de pegada a 250 m),
  `DataFrame: SensorLocalFrame`
- Já publicado em `OneDrive\Documents\AirSim\settings.json`

`tools/airsim_gt.py`
- `SELF_HIT_M = 2.0` — corta os ecos de fuselagem que poluíam as estatísticas
- `lidar_name` default → `"LidarNadir"`
- Novo campo `SensorSample.lidar_agl_m`: **mediana** (não média) dos alcances
  válidos = altura sobre a superfície dominante; observação independente de
  barômetro e GNSS, para entrar como fator de altura no FG
- Nova classe **`FrameRecorder`**: thread própria + **cliente RPC próprio**
  (capturar 1024×768 na thread da telemetria furaria os 20 Hz, e o socket
  msgpack não é thread-safe). Grava `frames/f_NNNNNN.jpg` (JPEG q90, ~100 kB),
  `clouds/c_NNNNNN.npy` (float32 N×3, já filtrado de zeros e self-hits) e o
  índice `frames.csv` com `t_wall` (mesma base do `telemetry.csv`) + GT do
  instante

`tools/run_gnss_denial_experiment.py`
- Coluna nova no CSV: `lidar_agl_m`
- Config/CLI: `--record-frames`, `--frame-hz` (4), `--cloud-hz` (2)
- Recorder inicia junto com o sampler e para antes do pouso; contagens vão para
  `meta.json` em `recording: {frames, clouds, capture_errors}`

`tools/run_gnss_campaign.py`
- Repassa `--record-frames/--frame-hz/--cloud-hz` para cada voo

Validado: `--dry-run` OK, JSON do settings OK, sem erros de lint.

## 7.5 Estado da bancada neste momento

- PX4 SITL **encerrado** (era resíduo da sessão anterior)
- CitySample **estava subindo** quando a sessão foi interrompida —
  **verificar/relançar antes de qualquer coisa**
- Campanha nova: **não rodou ainda**

```powershell
# 1. Subir a cena (esperar o mapa carregar de verdade)
Start-Process "D:\Projects\AirSim_Matrix\Windows\CitySample.exe" -ArgumentList "-windowed","-ResX=1280","-ResY=720"

# 2. Smoke test dos DOIS sensores novos antes de gastar campanha
.\venv\Scripts\python.exe tools\airsim_gt.py --check --vehicle PX4Drone --samples 5
#    esperar lidar_agl_m ~= altura sobre o solo (no spawn, poucos metros)
```

## 7.6 O que falta fazer (em ordem)

1. **Smoke test** com o simulador de pé: confirmar que `vio_cam` devolve imagem
   e que `LidarNadir` a 250 m devolve alcance ~250 m (não 0,6 m). Um voo curto
   com `--record-frames` já mostra se os JPEGs e `.npy` aparecem.
2. **Campanha nova** (o PX4 sobe sozinho via `px4_ensure_running`):
   ```powershell
   .\venv\Scripts\python.exe tools\run_gnss_campaign.py `
     --out-dir dataset_gnss_denial\campaign_vio250 `
     --alt-m 250 --distance-m 250 --deny-at-m 15 `
     --baseline-reps 2 --reps 3 --corridor-e 0 `
     --record-frames --frame-hz 4 --cloud-hz 2
   ```
   O baseline (GNSS ligado) serve de verdade de referência para avaliar o VIO e,
   se for usar terrain matching, de mapa de perfil de telhados.
3. **`tools/vio_frontend.py` — AINDA NÃO CRIADO.** Consome `frames.csv` de um
   run e produz `vio_odom.csv`. Desenho pretendido:
   - rastreio Shi-Tomasi + LK piramidal entre frames consecutivos
   - **seleção de keyframe por deslocamento**, não por tempo: a 250 m de altura
     um frame a 4 Hz com o veículo a ~6 m/s dá baseline/profundidade ≈ 1/100,
     o que **degenera a matriz essencial**. Acumular até ~20–40 m de baseline
     (razão 0,08–0,16) antes de fechar um par
   - `cv2.findEssentialMat` (RANSAC) + `recoverPose` → rotação relativa e
     **direção unitária** da translação; gravar também nº de inliers como peso
   - intrínsecos de FOV 90 em 1024×768: `fx = fy = 512`, `cx = 512`, `cy = 384`
4. **Estender `tools/run_factor_graph_replay.py`**: trocar o fator VO sintético
   pelo fator de **direção unitária** do `vio_odom.csv` (escala pela IMU) e
   adicionar **fator de altura via `lidar_agl_m`**. Manter o modo sintético como
   comparação, e o `note` do JSON deve deixar claro qual foi usado.
5. **Figura + abstract**: `figures/gnss_vio250_fg.*` e atualizar
   `wsc26/wsc26_gnss_abstract.tex` — hoje ele afirma que os fatores visuais são
   simulados; com VIO real essa frase **precisa mudar**. Manter 2 páginas.

## 7.7 Riscos conhecidos deste desenho

- **Parallax pequeno**: se a seleção de keyframe por deslocamento não bastar,
  reduzir FOV da `vio_cam` (mais resolução angular) ou aumentar o baseline.
- **Custo de captura**: 1024×768 a 4 Hz com `LockStep: true` pode injetar jitter
  no laço de 20 Hz. O recorder reancora o relógio em vez de acumular atraso, mas
  vale conferir `px4_age_s` e o nº de linhas do CSV contra os voos antigos.
- **Textura a 250 m**: 0,49 m/px. Telhados e ruas do CitySample devem dar
  features suficientes, mas não foi verificado ainda.
- As armadilhas da seção 4 continuam valendo, em especial o **timeout de param
  do MAVSDK em sessões longas** (o primeiro voo da sessão é o mais confiável).
