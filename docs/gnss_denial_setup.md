# Experimento GNSS-denied — Guia de bring-up

Medição da deriva da odometria inercial do EKF2 (PX4 SITL) contra o ground truth
do Cosys-AirSim, com negação de GNSS em voo.

Arquitetura: **PX4 SITL roda no WSL2** (não compila nativamente no Windows) e fala
lockstep com o **Cosys-AirSim no Windows** via TCP 4560. O script Python roda no
**Windows**, no venv do projeto, e fala MAVSDK/UDP 14540 com o PX4.

```
  Windows                                   WSL2 (Ubuntu)
  ┌──────────────────────────┐              ┌────────────────────┐
  │ Unreal + Cosys-AirSim    │◄── TCP 4560 ─┤ PX4 SITL           │
  │   (dinâmica de voo)      │   lockstep   │   (EKF2)           │
  │                          │◄─ UDP 14540 ─┤  link "Onboard"    │
  └──────────────────────────┘   (AirSim)   └────────────────────┘
             ▲                                        ▲
             │ RPC 41451                              │ UDP 14550
             │ (ground truth)                         │ (link "GCS")
             └──────────┬─────────────────────────────┘
                        │
              run_gnss_denial_experiment.py
```

### Por que 14550 e não 14540

O PX4 SITL sobe duas instâncias MAVLink:

```
mode: Normal,  udp port 18570 remote port 14550   <- livre, usamos esta
mode: Onboard, udp port 14580 remote port 14540   <- o AirSim faz bind aqui
```

Em SITL puro, `udp://:14540` é o caminho normal para o MAVSDK. **Com o AirSim no
circuito não é**: o `ControlPortLocal: 14540` do settings.json faz o AirSim tomar
essa porta e consumir os pacotes. O MAVSDK fica sem heartbeat e trava esperando.
Por isso o padrão do projeto é `udpin://0.0.0.0:14550`, o link de GCS.

---

## 1. Rede WSL2 (feito)

`C:\Users\andre\.wslconfig` já foi criado com `networkingMode=mirrored`. **Ainda
não está valendo** — precisa reiniciar o WSL:

```bash
wsl --shutdown
```

> Isso encerra a sessão Ubuntu em execução. Rode quando não houver trabalho aberto lá.

Validar depois de reiniciar:

```bash
wsl -e bash -lc "ip addr show | grep -c 'inet 127.0.0.1' && echo mirrored-ok"
```

---

## 2. Toolchain e build do PX4 (requer sua senha de sudo)

O `PX4-Autopilot` **já está clonado** em `~/PX4-Autopilot` no WSL (2.1 GB, branch `main`).
Falta instalar o toolchain e compilar — isso precisa de `sudo`, então rode você:

```bash
wsl -e bash -lc "cd ~/PX4-Autopilot && bash Tools/setup/ubuntu.sh --no-nuttx --no-sim-tools"
```

> **Risco conhecido:** seu WSL é **Ubuntu 26.04**, e o `ubuntu.sh` do PX4 é homologado
> até 24.04. Se falhar por dependência indisponível, o caminho mais rápido é instalar
> uma distro suportada em paralelo em vez de depurar pacotes:
> `wsl --install -d Ubuntu-24.04` e repetir o clone + setup lá.

Compilar o alvo **sem simulador embutido** (a dinâmica vem do AirSim):

```bash
wsl -e bash -lc "cd ~/PX4-Autopilot && make px4_sitl_default none_iris"
```

Primeira compilação leva ~10–20 min nos 24 vCPUs disponíveis.

---

## 3. Configuração do AirSim

Copiar o settings do experimento. **Nesta máquina o Documents é redirecionado para o
OneDrive**, então o caminho que o AirSim lê é `OneDrive\Documents\AirSim` — não
`%USERPROFILE%\Documents\AirSim`, que nem existe:

```powershell
Copy-Item config\cosys_airsim_px4_settings.json "$env:USERPROFILE\OneDrive\Documents\AirSim\settings.json"
```

> **Atenção:** isso substitui o settings usado pelos experimentos de detecção
> (exp30/multicam). Backup atual: `settings_final_ICEA.json` e `settings - Copy.json`
> na mesma pasta.

Para voltar ao config de detecção:

```powershell
Copy-Item "$env:USERPROFILE\OneDrive\Documents\AirSim\settings_final_ICEA.json" "$env:USERPROFILE\OneDrive\Documents\AirSim\settings.json"
```

Diferenças que importam nesse arquivo:

| Campo | Valor | Porquê |
|---|---|---|
| `VehicleType` | `PX4Multirotor` | Os outros configs usam `SimpleFlight`, que não tem estimador real |
| `ClockSpeed` | `1.0` | O config de detecção usa `5.0`; lockstep com PX4 exige tempo real |
| `LockStep` | `true` | Sincroniza passo de simulação com o PX4; sem isso o EKF diverge por jitter |
| `EKF2_HGT_REF` | `0` (Baro) | Default do PX4 é `1` (GPS) — ver seção 6 |

---

## 4. Ordem de inicialização

1. Inicie o ambiente Unreal e **espere o mapa carregar**. Ele fica aguardando a
   conexão do PX4 na TCP 4560.
   ```powershell
   & "D:\Projects\AirSim_Matrix\Windows\CitySample.exe" -windowed -ResX=1280 -ResY=720
   ```
2. Inicie o PX4 no WSL:
   ```bash
   wsl -e bash -lc "cd ~/PX4-Autopilot && PX4_SIM_HOSTNAME=127.0.0.1 make px4_sitl_default none_iris"
   ```
   Espere a linha `INFO [simulator_mavlink] Simulator connected`.
3. Rode o experimento no Windows (venv ativado).

---

## 5. Execução por estágios

Cada estágio só faz sentido se o anterior passou.

**Estágio 1 — sem simulador** (já validado ✅):
```bash
python tools\run_gnss_denial_experiment.py --dry-run
```

**Estágio 2 — ponte viva, sem armar:**
```bash
python tools\run_gnss_denial_experiment.py --no-fly --watch-s 10
```
Espera-se `global_pos_ok=True`, ~100 linhas no CSV e `err_norm` da ordem de 1 cm.

> **Na primeira execução isto vai abortar** com uma mensagem sobre `EKF2_HGT_REF`.
> É esperado — o parâmetro nasce em GPS e é `reboot_required`. O script grava o
> valor certo e para; reinicie o PX4 uma vez e rode de novo. O valor persiste no
> SITL, então isso acontece só uma vez por instalação.

**Estágio 3 — voo de referência, GNSS sempre ligado:**
```bash
python tools\run_gnss_denial_experiment.py --deny-at-m -1
```
`err_norm` deve ficar pequeno e **estável**. Se derivar aqui, o problema é
alinhamento de referencial ou lockstep — não o EKF.

**Estágio 4 — run experimental:**
```bash
python tools\run_gnss_denial_experiment.py --deny-at-m 25
```

**Estágio 5 — robustez:** `Ctrl+C` no meio do voo; o CSV deve ficar íntegro em disco.

---

## 6. Notas de método

**Por que `EKF2_HGT_REF` é ajustado antes do voo, e não junto com a negação.**
Esse parâmetro é `reboot_required` no PX4 — mudá-lo em voo não tem efeito. Se ele
ficar no default (`1` = GPS) e o `EKF2_GPS_CTRL` for zerado, o EKF fica sem
*nenhuma* fonte de altitude e a estimativa vertical colapsa, o que mediria uma
falha de configuração em vez da deriva inercial. Por isso `ensure_baro_height_ref()`
roda antes de armar e reinicia o PX4 se necessário.

**O que a negação faz.** `EKF2_GPS_CTRL` é um bitmask (default `7`: lat/lon +
altitude + velocidade 3D). Zerá-lo corta toda a fusão de GNSS imediatamente, sem
reboot, deixando X/Y em dead-reckoning inercial puro enquanto Z segue observável
pelo barômetro. O erro horizontal deve crescer de forma não-limitada — esse é o
resultado esperado, não um bug.

**Gatilho da negação.** Usa o ground truth (`cosys_x`), não a estimativa do PX4.
Se usasse a estimativa, o ponto de corte se deslocaria justamente por causa da
deriva sendo medida.

**Compensação de latência — importante para o artigo.** As duas fontes não são
lidas no mesmo instante: a amostra do PX4 chega por MAVLink e fica em cache, e o
ground truth é lido por RPC no instante do tick. Comparar as duas cruas transforma
essa defasagem em erro aparente **proporcional à velocidade**.

Isso foi medido, não suposto. No primeiro voo de referência (GNSS ligado, onde o
erro deveria ser plano) o erro era ~0 parado e crescia até 3,4 m em movimento, com
`err_x / v` travado em ~0,13–0,26 s — a assinatura inequívoca de defasagem
temporal, não de deriva do estimador. A 12 m/s isso é maior que a própria deriva
que o experimento quer observar.

Correção aplicada, em três partes:

1. `set_rate_position_velocity_ned(50 Hz)` — reduz o envelhecimento da amostra em
   cache. Efeito medido: `px4_age_s` caiu para ~2 ms (máx. 32 ms).
2. Com o cache já fresco, o erro **continuou** escalando com a velocidade, o que
   localizou a latência a montante: no caminho PX4 → MAVLink → `mavsdk_server` →
   processo, que o timestamp de chegada não enxerga.
3. Essa latência de transporte foi **medida**, não arbitrada: minimizando o RMS do
   erro sobre o voo de referência, τ = **255 ms** (RMS 1,54 m → 0,70 m). O valor é
   o default de `--latency-s` e o ground truth é retrocedido por
   `px4_age_s + τ` usando a velocidade verdadeira.

Reestimar τ numa bancada diferente (rodar num voo de **referência**, com GNSS ligado):

```bash
python tools\plot_gnss_drift.py dataset_gnss_denial\<run> --estimate-latency
```

> O estimador usa apenas o trecho com GNSS ativo. Rodá-lo num voo com GNSS negado
> faria o ajuste absorver a deriva real como se fosse atraso, inflando τ e
> escondendo o efeito que o experimento quer medir.

**Piso de medição.** Mesmo com τ corrigido, resta ~0,7 m de RMS no voo de
referência — soma do ruído de GPS simulado no EKF2, do erro de interpolação entre
amostras e da latência residual. Esse é o **piso de medição da bancada** e deve
ser citado como limitação: deriva abaixo dessa ordem não é distinguível. Para o
experimento isso é aceitável, já que a deriva sem GNSS chega a dezenas de metros.

O CSV guarda `px4_age_s` e `err_raw_norm` (erro sem compensação alguma) para que a
correção seja auditável, e não uma caixa-preta.

**Alinhamento de referencial.** A origem NED do PX4 é fixada na inicialização do
EKF; a do AirSim é o spawn do veículo. O CSV guarda os valores **crus** de cada
fonte; o offset medido na primeira amostra válida vai para `meta.json`
(`origin_offset_px4_to_airsim`) e é aplicado nas colunas `err_*` e nas figuras.

---

## 7. Saídas

```
dataset_gnss_denial/run_<timestamp>/
├── telemetry.csv   # 10 Hz, escrita incremental com flush (sobrevive a Ctrl+C)
└── meta.json       # offset de origem, health, params alterados, instante da negação

figures/
├── gnss_run_<timestamp>_trajectory_3d.{png,pdf}
├── gnss_run_<timestamp>_top_view.{png,pdf}
├── gnss_run_<timestamp>_error_vs_time.{png,pdf}   # figura principal do artigo
└── gnss_run_<timestamp>_summary.json              # métricas para citar no texto
```

Regerar figuras de um run existente sem voar de novo:

```bash
python tools\plot_gnss_drift.py dataset_gnss_denial\run_20260806_143000
```

---

## 8. Solução de problemas

### "EKF2 nao convergiu" logo na conexão

Quase sempre **não** é o EKF. Um `mavsdk_server.exe` órfão de um run anterior
continua com bind na 14550; o servidor novo não consegue a porta, não recebe nada,
e o timeout culpa o EKF. O script já mata servidores órfãos ao conectar, mas para
conferir manualmente:

```powershell
tasklist | Select-String mavsdk ; netstat -ano | Select-String ":14550"
```

Para confirmar que o EKF está de fato saudável, sem passar pelo orquestrador:

```bash
python tools\px4_link.py --check
```

### Falha ao armar no segundo voo da sessão

Depois de pousar, o PX4 permanece em modo `LAND` mesmo desarmado e no solo, e nesse
estado reporta `is_armable=False`. O `prepare_for_arm()` sai para `HOLD`
automaticamente. Se ainda assim não armar, reinicie o PX4 — é também a forma mais
limpa de garantir condições iniciais idênticas entre runs do artigo.

### O PX4 persiste `EKF2_GPS_CTRL` — a armadilha que mais custou tempo

**Sintoma:** depois do primeiro voo com negação de GNSS, *todos* os voos seguintes
falham na conexão, com `global_position_ok=False` ou nenhuma mensagem de `health`.
Reiniciar o PX4 não resolve. Reiniciar o Unreal não resolve. Trocar o
`settings.json` não resolve.

**Causa:** `deny_gnss()` grava `EKF2_GPS_CTRL=0`, e **o PX4 salva parâmetros em
disco** (`build/px4_sitl_default/rootfs/parameters.bson`). Todo boot seguinte sobe
com a fusão de GNSS desligada, então o EKF nunca converge. O experimento envenena
o próprio ambiente, e a mensagem de erro não aponta para a causa.

**Proteção no código, em duas camadas:**
- `shutdown()` restaura `EKF2_GPS_CTRL` **antes** de tentar pousar, porque o pouso
  pode falhar e não pode levar a restauração junto.
- `ensure_gnss_enabled()` roda no início de todo voo e restaura o valor se
  encontrar resíduo — com retry, porque logo após o boot a leitura de parâmetro dá
  timeout. Assim cada run é independente de o anterior ter encerrado bem.

**Recuperação manual**, se a bancada já estiver envenenada:

```bash
wsl -e bash -lc 'pkill -f "bin/px4"; rm -f ~/PX4-Autopilot/build/px4_sitl_default/rootfs/parameters*.bson'
```

Isso volta tudo ao default; o `EKF2_HGT_REF` também é perdido, e o
`ensure_baro_height_ref()` vai pedir um restart na próxima execução (comportamento
esperado, ver seção 5).

### Reiniciar o PX4 no meio da sessão quebra o lockstep

**Sintoma:** o primeiro voo da sessão funciona; do segundo em diante, todos falham
com "sem mensagens de health", mesmo com o PX4 imprimindo `Simulator connected`.

**Causa:** matar um PX4 com o lockstep **ativo** deixa o Cosys-AirSim preso à sessão
TCP anterior. O novo PX4 abre a conexão, mas não recebe dados de sensor, e o EKF
nunca converge.

**Consequência prática:** reiniciar o PX4 entre voos — o caminho óbvio para
condições iniciais idênticas — **só é seguro reiniciando o Unreal junto**, o que
custa minutos por voo.

Por isso `run_gnss_campaign.py` **não reinicia por padrão**. Cada voo já é
independente do anterior nos aspectos que afetam a medida:
`ensure_gnss_enabled()` restaura o parâmetro de GNSS, `prepare_for_arm()` sai do
modo `LAND` residual, e a deriva é medida a partir de `t_since_denial`, não da
posição absoluta. O que carrega entre voos é o estado interno do EKF, já
convergido com GNSS ativo antes de cada negação.

O flag `--restart-px4` existe para quem quiser pagar o custo, mas exige reiniciar
o simulador junto.

### Nunca use `client.reset()` com veículo PX4

Medido nesta bancada: depois de `simClient.reset()`, o PX4 fica permanentemente em
`local_position_ok=True` mas `global_position_ok=False` e `home_position_ok=False`,
e o voo nunca arma. **Reiniciar o PX4 não recupera** — o dano é do lado do AirSim,
e só um restart do Unreal resolve.

Por isso a campanha (`run_gnss_campaign.py`) reinicia **apenas o PX4** entre voos.
Isso já entrega o que importa: um EKF limpo a cada repetição. A posição inicial
varia entre voos, mas a deriva é medida a partir do instante da negação
(`t_since_denial`), então não confunde a medida. O flag `--reset-airsim` existe
mas está desligado por padrão e não é recomendado.

### O AirSim trava esperando conexão

Com `LockStep: true` o simulador bloqueia até o PX4 conectar na TCP 4560. Isso é
esperado: suba o Unreal primeiro, o PX4 depois.

---

## 9. Fora de escopo (decidido)

- **Odometria visual (VIO).** Esta fase mede só dead-reckoning inercial, que é o
  baseline necessário antes de demonstrar qualquer ganho de VIO.
- Integração com o pipeline de detecção YOLO / exp30.
- Voos multi-veículo.
