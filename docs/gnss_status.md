# GNSS-denied — estado atual e pontos para retomar

Snapshot de 07/08/2026 01:10. Complementa `docs/gnss_denial_setup.md`.

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

## 5. Próximo (opcional)

- Mais denied reps se quiser apertar a banda p05–p95
- Factor-graph no mesmo CSV (trabalho futuro do abstract)
- Commit do que está untracked (só se pedido)
