"""
px4_link.py — Camada MAVSDK para o experimento GNSS-denied
==========================================================
Encapsula a conexao com o PX4 SITL (rodando no WSL2, ligado ao Cosys-AirSim via
lockstep TCP:4560) e expoe comandos de voo + negacao de GNSS em tempo de voo.

Este modulo NAO fala com o AirSim (ver airsim_gt.py) e NAO escreve CSV
(ver run_gnss_denial_experiment.py). Ele so controla e le o PX4.

Dependencias:
    pip install mavsdk

Uso (normalmente importado, mas roda standalone como smoke test):
    # Conecta, imprime health/EKF e sai (sem armar)
    python tools/px4_link.py --check

    # Conecta e loga odometria por 10 s sem voar
    python tools/px4_link.py --check --watch-s 10

Coordenadas NED (mesma convencao do AirSim): X=Norte, Y=Leste, Z=para baixo.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass, field

# MAVSDK 3.x: o formato antigo "udp://:14540" esta deprecado; use "udpin://host:porta".
#
# Porta 14550, nao 14540. O PX4 SITL sobe duas instancias MAVLink:
#   mode: Normal   udp 18570 -> remote 14550   (link "GCS")
#   mode: Onboard  udp 14580 -> remote 14540   (link "offboard")
# Em SITL puro o 14540 e o caminho normal para MAVSDK, mas com o AirSim no
# circuito ele e quem faz bind em 14540 (ControlPortLocal do settings.json) e
# consome esses pacotes. Sobra o link de GCS em 14550 para o nosso cliente.
DEFAULT_SYSTEM_ADDRESS = "udpin://0.0.0.0:14550"

# EKF2_GPS_CTRL e um bitmask (PX4 >= v1.14, default 7). Em versoes antigas o
# controle equivalente era EKF2_AID_MASK (bit 0 = GPS). Tentamos os dois.
GPS_CTRL_PARAM = "EKF2_GPS_CTRL"
GPS_AID_MASK_PARAM = "EKF2_AID_MASK"
# Bitmask default do PX4: lon/lat + altitude + velocidade 3D.
GPS_CTRL_DEFAULT = 7

# EKF2_HGT_REF: 0=Baro, 1=GPS, 2=Range, 3=Vision. Default do PX4 = 1 (GPS).
# ATENCAO: este parametro e reboot_required=true no PX4 — precisa ser ajustado
# ANTES de armar, com reboot, e nao no momento da negacao (ver ensure_baro_height_ref).
HGT_REF_PARAM = "EKF2_HGT_REF"
HGT_REF_BARO = 0
HGT_REF_GPS = 1


@dataclass
class OdometrySample:
    """Ultima estimativa de posicao local do EKF2 (NED, metros)."""
    t_wall: float = 0.0
    north_m: float = float("nan")
    east_m: float = float("nan")
    down_m: float = float("nan")
    vn_ms: float = float("nan")
    ve_ms: float = float("nan")
    vd_ms: float = float("nan")
    valid: bool = False


@dataclass
class PX4Link:
    """Conexao MAVSDK com o PX4 SITL."""

    system_address: str = DEFAULT_SYSTEM_ADDRESS
    takeoff_alt_m: float = 10.0

    drone: object | None = field(default=None, init=False)
    latest: OdometrySample = field(default_factory=OdometrySample, init=False)
    gps_denied: bool = field(default=False, init=False)
    denial_t_wall: float | None = field(default=None, init=False)
    _tasks: list = field(default_factory=list, init=False)
    _offboard_started: bool = field(default=False, init=False)

    # ─── Conexao ─────────────────────────────────────────────────────────────

    @staticmethod
    def _kill_stale_servers() -> None:
        """
        Mata mavsdk_server orfaos de um run anterior.

        Cada System() sobe seu proprio mavsdk_server, que faz bind na porta UDP. Se
        um run morre por excecao ou Ctrl+C, o servidor pode sobreviver e continuar
        segurando a porta; o run seguinte sobe outro servidor que nao consegue bind,
        nao recebe nada, e falha com "EKF2 nao convergiu" — um sintoma que nao tem
        nada a ver com o EKF. Limpar antes evita esse falso diagnostico.
        """
        import subprocess
        try:
            subprocess.run(
                ["taskkill", "/F", "/IM", "mavsdk_server.exe"],
                capture_output=True, timeout=10,
            )
        except Exception:
            pass  # Nao-Windows ou nenhum processo: seguir normalmente.

    async def connect(self, timeout_s: float = 120.0) -> None:
        """Conecta ao PX4 e espera o EKF2 convergir (posicao global + home)."""
        from mavsdk import System

        self._kill_stale_servers()
        print(f"[INFO] Conectando ao PX4 em {self.system_address} ...")
        self.drone = System()
        await self.drone.connect(system_address=self.system_address)

        deadline = time.monotonic() + timeout_s
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("[OK] PX4 conectado (heartbeat recebido).")
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Sem heartbeat do PX4 apos {timeout_s:.0f}s em {self.system_address}. "
                    "Verifique se o PX4 SITL esta rodando e se o .wslconfig usa networkingMode=mirrored."
                )

        # ANTES de esperar o EKF: garantir que a fusao de GNSS esta ligada.
        # Sem isto o wait abaixo nunca termina — ver ensure_gnss_enabled().
        await self.ensure_gnss_enabled()

        print("[INFO] Aguardando convergencia do EKF2 (global position + home) ...")

        async def _wait_ekf() -> None:
            last = None
            async for health in self.drone.telemetry.health():
                if health.is_global_position_ok and health.is_home_position_ok:
                    print("[OK] EKF2 convergido.")
                    return
                last = health
                if time.monotonic() > deadline:
                    break
            raise TimeoutError(
                "EKF2 nao convergiu (global_pos_ok="
                f"{getattr(last, 'is_global_position_ok', '?')}, home_pos_ok="
                f"{getattr(last, 'is_home_position_ok', '?')}). Se local_position_ok=True mas "
                "global=False, o GPS nao reinicializou — tipico apos client.reset() do AirSim "
                "num veiculo PX4. Reinicie o PX4 sem resetar o simulador."
            )

        # wait_for e necessario: se o stream de health parar de emitir, o `async for`
        # bloqueia para sempre e a checagem de deadline dentro dele nunca roda.
        try:
            await asyncio.wait_for(_wait_ekf(), timeout=timeout_s)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Sem mensagens de health do PX4 apos {timeout_s:.0f}s. "
                "Verifique se ha um mavsdk_server orfao segurando a porta."
            ) from None

    async def health_report(self) -> dict:
        """Snapshot de health + parametros relevantes do EKF, para diagnostico."""
        async for h in self.drone.telemetry.health():
            report = {
                "gyro_cal_ok": h.is_gyrometer_calibration_ok,
                "accel_cal_ok": h.is_accelerometer_calibration_ok,
                "mag_cal_ok": h.is_magnetometer_calibration_ok,
                "local_pos_ok": h.is_local_position_ok,
                "global_pos_ok": h.is_global_position_ok,
                "home_pos_ok": h.is_home_position_ok,
                "armable": h.is_armable,
            }
            break
        for name in (GPS_CTRL_PARAM, GPS_AID_MASK_PARAM, HGT_REF_PARAM):
            try:
                report[name] = await self.drone.param.get_param_int(name)
            except Exception:
                report[name] = None  # parametro nao existe nesta versao do PX4
        return report

    async def ensure_gnss_enabled(self) -> dict:
        """
        Restaura a fusao de GNSS antes do voo.

        O PX4 PERSISTE parametros. Como deny_gnss() grava EKF2_GPS_CTRL=0, um voo
        anterior (ou um run que morreu antes de restaurar) deixa a bancada com GNSS
        permanentemente desligado: todo boot seguinte sobe com o EKF sem aidingo,
        `global_position_ok` nunca fica True, e a conexao falha por timeout com uma
        mensagem que nao aponta para a causa. Esta checagem torna cada run
        independente do anterior, em vez de depender do shutdown ter corrido bem.
        """
        applied: dict = {}
        # Logo apos o boot o subsistema de parametros ainda nao responde e a
        # leitura da TIMEOUT. Nao pode ser fatal nem ser ignorado: se desistirmos
        # na primeira falha, um EKF2_GPS_CTRL=0 residual passa despercebido e o
        # voo inteiro sai invalido. Por isso insiste antes de desistir.
        # Aqui NAO baixamos a telemetria: ensure_gnss_enabled() roda no connect,
        # ANTES do wait de health. Derrubar o stream nesse momento foi associado a
        # sessões em que o heartbeat chega e o health nao (campanha apos smoke).
        current = None
        for attempt in range(6):
            try:
                current = await self.drone.param.get_param_int(GPS_CTRL_PARAM)
                break
            except Exception as e:
                if attempt == 5:
                    print(f"[AVISO] Nao foi possivel ler {GPS_CTRL_PARAM} apos 6 tentativas: {e}")
                    return applied
                await asyncio.sleep(2.0)

        applied[f"{GPS_CTRL_PARAM}_found"] = current
        if current == GPS_CTRL_DEFAULT:
            return applied

        print(f"[INFO] {GPS_CTRL_PARAM}={current} (residuo de um voo anterior). "
              f"Restaurando para {GPS_CTRL_DEFAULT} ...")
        await self.drone.param.set_param_int(GPS_CTRL_PARAM, GPS_CTRL_DEFAULT)
        applied[f"{GPS_CTRL_PARAM}_restored_to"] = GPS_CTRL_DEFAULT
        await asyncio.sleep(2.0)  # deixa o EKF reagir ao aiding voltando
        self.gps_denied = False
        return applied

    async def ensure_baro_height_ref(self) -> dict:
        """
        Garante EKF2_HGT_REF=Baro ANTES do voo. Se precisar mudar, ABORTA o voo.

        Por que aqui e nao em deny_gnss(): EKF2_HGT_REF e reboot_required no PX4.
        Se ele ficar em GPS (o default) e depois zerarmos EKF2_GPS_CTRL em voo, o
        EKF fica sem nenhuma fonte de altitude e a estimativa vertical colapsa —
        mediriamos uma falha de configuracao, nao a deriva inercial que e o objeto
        do estudo. Com Baro como referencia, o corte de GNSS deixa X/Y em
        dead-reckoning puro enquanto Z segue observavel, que e o cenario pretendido.

        Por que abortar em vez de reiniciar sozinho: o PX4 nega `reboot` por MAVLink
        (COMMAND_DENIED) neste estado, e reconectar exigiria derrubar o mavsdk_server
        que ainda segura a porta UDP. Como o parametro persiste no SITL, setar +
        reiniciar o PX4 uma vez resolve de forma definitiva. Voar com a referencia
        errada produziria dado invalido em silencio, entao o certo e parar aqui.
        """
        applied: dict = {}
        try:
            current = await self.drone.param.get_param_int(HGT_REF_PARAM)
        except Exception as e:
            print(f"[AVISO] Nao foi possivel ler {HGT_REF_PARAM}: {e}")
            return applied

        applied[f"{HGT_REF_PARAM}_before"] = current
        if current == HGT_REF_BARO:
            print(f"[OK] {HGT_REF_PARAM} ja esta em Baro.")
            applied[f"{HGT_REF_PARAM}_after"] = HGT_REF_BARO
            applied["needs_restart"] = False
            return applied

        print(f"[INFO] {HGT_REF_PARAM}={current} (GPS). Gravando Baro ...")
        await self.drone.param.set_param_int(HGT_REF_PARAM, HGT_REF_BARO)
        applied[f"{HGT_REF_PARAM}_after"] = HGT_REF_BARO
        applied["needs_restart"] = True

        raise RuntimeError(
            f"{HGT_REF_PARAM} foi gravado como Baro, mas so vale apos reiniciar o PX4 "
            "(parametro reboot_required). O valor persiste no SITL, entao basta "
            "reiniciar o PX4 uma vez e rodar de novo:\n"
            "    cd ~/PX4-Autopilot && PX4_SIM_HOSTNAME=127.0.0.1 make px4_sitl_default none_iris"
        )

    # ─── Stream de odometria ─────────────────────────────────────────────────

    async def _pump_odometry(self) -> None:
        """Task de fundo: mantem self.latest com a ultima amostra do EKF2."""
        async for pv in self.drone.telemetry.position_velocity_ned():
            self.latest = OdometrySample(
                t_wall=time.time(),
                north_m=pv.position.north_m,
                east_m=pv.position.east_m,
                down_m=pv.position.down_m,
                vn_ms=pv.velocity.north_m_s,
                ve_ms=pv.velocity.east_m_s,
                vd_ms=pv.velocity.down_m_s,
                valid=True,
            )

    async def set_stream_rate(self, hz: float = 25.0) -> None:
        """
        Pede ao PX4 uma taxa para position_velocity_ned.

        A amostra em cache envelhece ate 1/taxa antes de ser lida. Como o erro que
        estamos medindo e comparado contra um ground truth lido no instante do tick,
        essa idade vira erro aparente proporcional a velocidade (v * idade). A 5 Hz
        e 12 m/s isso daria ~2,4 m de erro puramente artificial.

        25 Hz e nao 50: a 50 Hz o link de GCS saturava e o cliente de parametros do
        MAVSDK passava a falhar ("retrying failed"), derrubando a negacao de GNSS —
        que e o momento critico do experimento. Como amostramos a 20 Hz e a idade
        medida do cache ficou em ~2 ms (a latencia real e de transporte, nao de
        cache), 25 Hz nao perde nada e devolve banda para os parametros.
        """
        try:
            await self.drone.telemetry.set_rate_position_velocity_ned(hz)
            print(f"[OK] Taxa de position_velocity_ned pedida: {hz:.0f} Hz.")
        except Exception as e:
            print(f"[AVISO] Nao foi possivel elevar a taxa do stream ({e}); seguindo com o padrao.")

    def start_streams(self) -> None:
        """Dispara as tasks de fundo que alimentam o cache de odometria."""
        self._tasks.append(asyncio.create_task(self._pump_odometry()))

    async def stop_streams(self) -> None:
        for t in self._tasks:
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    # ─── Comandos de voo ─────────────────────────────────────────────────────

    async def prepare_for_arm(self, timeout_s: float = 45.0) -> None:
        """
        Deixa o veiculo armavel, saindo de um modo LAND residual.

        Depois de um voo, o PX4 fica em LAND mesmo apos tocar o solo e desarmar, e
        nesse estado recusa armar (is_armable=False). Sem isto, o segundo run da
        sessao falha no arm sem motivo aparente.
        """
        async for fm in self.drone.telemetry.flight_mode():
            mode = str(fm)
            break
        async for h in self.drone.telemetry.health():
            if h.is_armable:
                return
            break

        print(f"[INFO] Veiculo nao armavel (modo {mode}); saindo para HOLD ...")
        try:
            await self.drone.action.hold()
        except Exception as e:
            print(f"[AVISO] hold() falhou: {e}")

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            async for h in self.drone.telemetry.health():
                if h.is_armable:
                    print("[OK] Veiculo armavel.")
                    return
                break
            await asyncio.sleep(0.5)
        raise RuntimeError(
            "Veiculo nao ficou armavel. Reinicie o PX4 SITL para voltar ao estado inicial:\n"
            "    cd ~/PX4-Autopilot && PX4_SIM_HOSTNAME=127.0.0.1 make px4_sitl_default none_iris"
        )

    async def arm_and_takeoff(self, east_m: float | None = None) -> None:
        """
        Arma, decola e sobe ate takeoff_alt_m por setpoint offboard.

        O `takeoff` do PX4 sozinho nao e confiavel como referencia de altitude aqui:
        medido nesta bancada, o EKF estabilizava perto de 7 m com 10 m comandados e
        a espera por 90% do alvo consumia os 60 s de timeout inteiros. Isso fazia a
        duracao dos voos variar em 2,3x entre repeticoes, inviabilizando compara-las.
        Subir por offboard ate o alvo e deterministico e deixa a fase de subida com
        duracao previsivel.
        """
        await self.prepare_for_arm()
        print("[INFO] Armando ...")
        await self.drone.action.arm()
        await self.drone.action.set_takeoff_altitude(self.takeoff_alt_m)
        print(f"[INFO] Decolando ...")
        await self.drone.action.takeoff()

        # Sai do chao antes de assumir offboard (offboard no solo e recusado).
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if self.latest.valid and -self.latest.down_m >= 2.0:
                break
            await asyncio.sleep(0.2)

        await self.start_offboard()
        await self.climb_to(self.takeoff_alt_m, east_m=east_m)

    async def climb_to(self, alt_m: float, tol_m: float = 1.0,
                       timeout_s: float = 120.0, east_m: float | None = None) -> None:
        """
        Sobe ate alt_m por setpoint offboard e espera estabilizar.

        `east_m` opcional posiciona o veiculo no corredor validado (ver
        probe_gnss_corridor.py) durante a subida, antes do trecho reto comecar.
        """
        from mavsdk.offboard import PositionNedYaw

        n = self.latest.north_m
        e = self.latest.east_m if east_m is None else east_m
        alvo = f"{alt_m:.0f} m" + (f", corredor E={e:.0f}" if east_m is not None else "")
        print(f"[INFO] Subindo para {alvo} por offboard ...")
        await self.drone.offboard.set_position_ned(PositionNedYaw(n, e, -alt_m, 0.0))

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if (self.latest.valid
                    and abs(-self.latest.down_m - alt_m) <= tol_m
                    and (east_m is None or abs(self.latest.east_m - e) <= 3.0)):
                print(f"[OK] Altitude {(-self.latest.down_m):.1f} m, E={self.latest.east_m:.1f}.")
                return
            await asyncio.sleep(0.2)
        print(f"[AVISO] Altitude alvo nao atingida em {timeout_s:.0f}s "
              f"(atual {(-self.latest.down_m):.1f} m); seguindo.")

    async def start_offboard(self) -> None:
        """Entra em modo offboard mantendo a posicao atual (setpoint obrigatorio antes de start)."""
        from mavsdk.offboard import OffboardError, PositionNedYaw

        hold = PositionNedYaw(
            self.latest.north_m, self.latest.east_m, self.latest.down_m, 0.0
        )
        await self.drone.offboard.set_position_ned(hold)
        try:
            await self.drone.offboard.start()
            self._offboard_started = True
            print("[OK] Modo offboard ativo.")
        except OffboardError as e:
            raise RuntimeError(f"Falha ao entrar em offboard: {e._result.result}") from e

    async def fly_straight_x(
        self,
        distance_m: float = 50.0,
        alt_m: float | None = None,
        arrive_tol_m: float = 1.5,
        timeout_s: float = 180.0,
        direction: int = 1,
        east_m: float | None = None,
    ):
        """
        Voa em linha reta no eixo X por `distance_m` a partir da posicao atual.

        Usa setpoint de posicao offboard (nao goto_location): trajetoria reta e
        deterministica em NED local, que e o referencial do estudo.

        `direction` = +1 (Norte) ou -1 (Sul). O vaivem faz o veiculo repetir sempre
        o MESMO trecho de cidade em vez de avancar para regiao nova a cada voo — o
        que evitava tanto o acumulo de posicao quanto entrar em area com obstaculos
        nao validados, que foi a origem das colisoes observadas.

        Retorna (start_north, target_north) para o orquestrador saber o progresso.
        """
        from mavsdk.offboard import PositionNedYaw

        alt = alt_m if alt_m is not None else self.takeoff_alt_m
        start_n = self.latest.north_m
        start_e = self.latest.east_m if east_m is None else east_m
        d = 1 if direction >= 0 else -1
        target_n = start_n + d * distance_m
        eixo = "+X" if d > 0 else "-X"

        print(f"[INFO] Voando {distance_m:.1f} m em {eixo} (N {start_n:.1f} -> {target_n:.1f}) ...")
        # Yaw acompanha o sentido: voar de re com o nariz fixo mudaria o perfil de
        # excitacao da IMU entre idas e voltas, e a deriva inercial depende disso.
        yaw = 0.0 if d > 0 else 180.0
        await self.drone.offboard.set_position_ned(
            PositionNedYaw(target_n, start_e, -alt, yaw)
        )

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.latest.valid and abs(self.latest.north_m - target_n) <= arrive_tol_m:
                print("[OK] Ponto final alcancado (segundo o EKF2).")
                return start_n, target_n
            await asyncio.sleep(0.1)
        print("[AVISO] Timeout no trecho reto; seguindo para o encerramento.")
        return start_n, target_n

    # ─── Negacao de GNSS ─────────────────────────────────────────────────────

    async def _with_low_telemetry_for_params(self) -> None:
        """
        Baixa a taxa do stream antes de mexer em parametros.

        Medido: com position_velocity_ned a 25 Hz o cliente de parametros do MAVSDK
        falha com TIMEOUT durante o voo (o smoke a 250 m chegou ao fim sem negacao).
        Em solo, logo apos o boot, a leitura as vezes passa; em movimento o link
        satura. Diminuir o stream por alguns segundos devolve banda ao PARAM_*.
        """
        try:
            await self.drone.telemetry.set_rate_position_velocity_ned(2.0)
        except Exception:
            pass
        await asyncio.sleep(1.0)

    async def _restore_telemetry_rate(self, hz: float = 25.0) -> None:
        try:
            await self.drone.telemetry.set_rate_position_velocity_ned(hz)
        except Exception:
            pass

    async def deny_gnss(self) -> dict:
        """
        Corta a fusao de GNSS no EKF2, deixando o estimador em dead-reckoning inercial.

        So mexe em EKF2_GPS_CTRL (bitmask, default 7), que vale imediatamente sem
        reboot. A referencia de altura ja deve ter sido movida para barometro por
        ensure_baro_height_ref() antes do voo — la esta a explicacao do porque.

        Retorna dict com o que foi efetivamente alterado (vai para o meta.json).
        """
        applied: dict = {}

        # PX4 novo usa EKF2_GPS_CTRL; versoes antigas, EKF2_AID_MASK (bit 0 = GPS).
        #
        # Retry e obrigatorio aqui: o subsistema de parametros do PX4 da TIMEOUT
        # esporadico, e este e o momento mais critico do experimento. Sem insistir,
        # a negacao falha silenciosamente, `gps_denied` fica False, o gatilho
        # redispara a cada tick e o voo inteiro sai sem negacao nenhuma — dado
        # invalido que parece valido.
        await self._with_low_telemetry_for_params()
        denied_via = None
        last_err: Exception | None = None
        before = GPS_CTRL_DEFAULT
        try:
            for attempt in range(8):
                try:
                    # Escrita primeiro: o get preliminar tambem TIMEOUT e queimava
                    # as tentativas antes de qualquer set. Confirmamos depois.
                    await self.drone.param.set_param_int(GPS_CTRL_PARAM, 0)
                    await asyncio.sleep(0.3)
                    confirmed = await self.drone.param.get_param_int(GPS_CTRL_PARAM)
                    if confirmed != 0:
                        raise RuntimeError(
                            f"{GPS_CTRL_PARAM} continua {confirmed} apos a escrita"
                        )
                    applied[f"{GPS_CTRL_PARAM}_before"] = before
                    applied[f"{GPS_CTRL_PARAM}_after"] = 0
                    applied["attempts"] = attempt + 1
                    denied_via = GPS_CTRL_PARAM
                    break
                except Exception as e:
                    last_err = e
                    if attempt < 7:
                        # Backoff crescente: o timeout do cliente de parametros do
                        # MAVSDK aparece em rajadas, e insistir na mesma cadencia
                        # so consome as tentativas dentro da mesma rajada. Esperar
                        # progressivamente mais da chance de a fila drenar.
                        # Atrasar a negacao nao invalida o voo: o instante real e
                        # registrado e a deriva e medida a partir dele.
                        await asyncio.sleep(1.0 + 1.5 * attempt)

            if denied_via is None:
                # Fallback so para PX4 antigo. Se GPS_CTRL existe e falhou, o problema
                # e de comunicacao, nao de versao — relatar o erro original.
                try:
                    before_m = await self.drone.param.get_param_int(GPS_AID_MASK_PARAM)
                    after = before_m & ~1  # limpa o bit 0 (GPS)
                    await self.drone.param.set_param_int(GPS_AID_MASK_PARAM, after)
                    applied[f"{GPS_AID_MASK_PARAM}_before"] = before_m
                    applied[f"{GPS_AID_MASK_PARAM}_after"] = after
                    denied_via = GPS_AID_MASK_PARAM
                except Exception:
                    raise RuntimeError(
                        f"Nao foi possivel negar GNSS apos 8 tentativas em {GPS_CTRL_PARAM}: "
                        f"{last_err}"
                    ) from last_err
        finally:
            await self._restore_telemetry_rate(25.0)

        self.gps_denied = True
        self.denial_t_wall = time.time()
        applied["denied_via"] = denied_via
        applied["denial_t_wall"] = self.denial_t_wall
        print(f"[OK] GNSS negado via {denied_via}. EKF2 em dead-reckoning inercial.")
        return applied

    # ─── Encerramento ────────────────────────────────────────────────────────

    async def shutdown(self, land: bool = True) -> None:
        """
        Encerramento tolerante a falha: cada etapa e independente, para que um erro
        no offboard nao impeça o pouso, nem um erro no pouso impeça parar as streams.
        """
        # Restaurar a fusao de GNSS PRIMEIRO: o PX4 persiste parametros, entao
        # sair daqui com EKF2_GPS_CTRL=0 deixa a bancada inutilizavel para o
        # proximo run. Vem antes do pouso porque pousar pode falhar.
        if self.gps_denied:
            try:
                await self.drone.param.set_param_int(GPS_CTRL_PARAM, GPS_CTRL_DEFAULT)
                self.gps_denied = False
                print(f"[OK] {GPS_CTRL_PARAM} restaurado para {GPS_CTRL_DEFAULT}.")
            except Exception as e:
                print(f"[AVISO] Falha ao restaurar {GPS_CTRL_PARAM}: {e}. "
                      "O proximo run corrige via ensure_gnss_enabled().")

        if self._offboard_started:
            try:
                await self.drone.offboard.stop()
            except Exception as e:
                print(f"[AVISO] offboard.stop falhou: {e}")
        if land:
            try:
                print("[INFO] Pousando ...")
                await self.drone.action.land()
            except Exception as e:
                print(f"[AVISO] land falhou: {e}")
        await self.stop_streams()


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test standalone
# ─────────────────────────────────────────────────────────────────────────────

async def _check(args) -> None:
    link = PX4Link(system_address=args.system_address)
    await link.connect(timeout_s=args.timeout_s)
    link.start_streams()

    report = await link.health_report()
    print("\n[HEALTH]")
    for k, v in report.items():
        print(f"  {k:24s} = {v}")

    if args.watch_s > 0:
        print(f"\n[INFO] Observando odometria por {args.watch_s:.0f} s a 10 Hz ...")
        t_end = time.monotonic() + args.watch_s
        while time.monotonic() < t_end:
            s = link.latest
            print(f"  N={s.north_m:8.2f}  E={s.east_m:8.2f}  D={s.down_m:8.2f}  valid={s.valid}")
            await asyncio.sleep(0.1)

    await link.stop_streams()


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test da conexao MAVSDK com o PX4 SITL")
    ap.add_argument("--check", action="store_true", help="Conecta e imprime health (nao arma)")
    ap.add_argument("--system-address", default=DEFAULT_SYSTEM_ADDRESS, metavar="URL")
    ap.add_argument("--timeout-s", type=float, default=60.0, metavar="S")
    ap.add_argument("--watch-s", type=float, default=0.0, metavar="S",
                    help="Loga odometria por N segundos apos o health")
    args = ap.parse_args()

    if not args.check:
        ap.error("use --check (este modulo e normalmente importado, nao executado)")
    asyncio.run(_check(args))


if __name__ == "__main__":
    main()
