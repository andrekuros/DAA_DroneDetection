"""
probe_gnss_corridor.py — Valida o corredor de voo antes de voar
================================================================
Sonda a cena com `simTestLineOfSightBetweenPoints` para descobrir em que altitude
(e com que deslocamento lateral) o trecho reto do experimento fica livre de
obstaculos, sem precisar bater neles para descobrir.

Motivacao: a 60 m no CitySample o veiculo colidia com torres, cada impacto
contamina a janela de deriva medida, e um voo que termina em colisao deixa o
proximo comecando ja encostado no obstaculo. Validar antes sai muito mais barato.

Dependencias:
    pip install cosysairsim

Uso:
    # Varre altitudes de 40 a 200 m no corredor padrao (250 m em +X do drone)
    python tools/probe_gnss_corridor.py

    # Corredor especifico, testando tambem desvios laterais
    python tools/probe_gnss_corridor.py --distance-m 250 --lateral-m -20 0 20

    # So checa uma altitude
    python tools/probe_gnss_corridor.py --altitudes 60
"""

from __future__ import annotations

import argparse
import math

try:
    import cosysairsim as airsim
    HAS_AIRSIM = True
except ImportError:
    try:
        import airsim
        HAS_AIRSIM = True
    except ImportError:
        HAS_AIRSIM = False

EARTH_M_PER_DEG = 111320.0


class NedGeo:
    """
    Conversao NED <-> geodesica por aproximacao de terra plana.

    Suficiente aqui: o corredor tem centenas de metros, escala em que o erro da
    aproximacao e muito menor que o tamanho dos obstaculos sondados.
    """

    def __init__(self, client, vehicle: str):
        env = client.simGetGroundTruthEnvironment(vehicle_name=vehicle)
        k = client.simGetGroundTruthKinematics(vehicle_name=vehicle)
        gp, p = env.geo_point, k.position
        # Referencia correspondente a NED (0,0,0)
        self.lat0 = gp.latitude - p.x_val / EARTH_M_PER_DEG
        self.lon0 = gp.longitude - p.y_val / (EARTH_M_PER_DEG * math.cos(math.radians(gp.latitude)))
        self.alt0 = gp.altitude + p.z_val  # z_ned e para baixo
        self.coslat = math.cos(math.radians(gp.latitude))

    def to_geo(self, north_m: float, east_m: float, down_m: float):
        g = airsim.GeoPoint()
        g.latitude = self.lat0 + north_m / EARTH_M_PER_DEG
        g.longitude = self.lon0 + east_m / (EARTH_M_PER_DEG * self.coslat)
        g.altitude = self.alt0 - down_m
        return g


def probe_corridor(client, conv: NedGeo, start_n: float, start_e: float,
                   distance_m: float, alt_m: float, direction: int,
                   step_m: float = 10.0) -> tuple[bool, float | None]:
    """
    Testa visada livre ao longo do corredor, em passos.

    Testar so extremo-a-extremo esconderia um obstaculo no meio quando a linha
    passa raspando por fora dele; segmentar encontra o primeiro bloqueio e diz
    ONDE ele esta, que e o que permite escolher um desvio.

    Retorna (livre, primeiro_n_bloqueado).
    """
    n_steps = max(1, int(distance_m / step_m))
    prev = conv.to_geo(start_n, start_e, -alt_m)
    for i in range(1, n_steps + 1):
        n = start_n + direction * (i * distance_m / n_steps)
        cur = conv.to_geo(n, start_e, -alt_m)
        try:
            if not client.simTestLineOfSightBetweenPoints(prev, cur):
                return False, n
        except Exception as e:
            print(f"[AVISO] LOS falhou em N={n:.0f}: {e}")
            return False, n
        prev = cur
    return True, None


def main() -> None:
    ap = argparse.ArgumentParser(description="Valida o corredor de voo do experimento GNSS")
    ap.add_argument("--vehicle", default="PX4Drone", metavar="NOME")
    ap.add_argument("--ip", default="127.0.0.1", metavar="IP")
    ap.add_argument("--distance-m", type=float, default=250.0, metavar="M")
    ap.add_argument("--direction", type=int, default=1, choices=[1, -1])
    ap.add_argument("--altitudes", type=float, nargs="+",
                    default=[40, 60, 80, 100, 120, 150, 180, 200], metavar="M")
    ap.add_argument("--lateral-m", type=float, nargs="+", default=[0.0], metavar="M",
                    help="Deslocamentos em Y a testar (procura uma rua livre)")
    ap.add_argument("--step-m", type=float, default=10.0, metavar="M")
    args = ap.parse_args()

    if not HAS_AIRSIM:
        ap.error("cosysairsim nao instalado")

    client = airsim.MultirotorClient(ip=args.ip)
    client.confirmConnection()
    conv = NedGeo(client, args.vehicle)
    k = client.simGetGroundTruthKinematics(vehicle_name=args.vehicle)
    start_n, start_e = k.position.x_val, k.position.y_val

    print(f"Origem do corredor: N={start_n:.1f} E={start_e:.1f}, "
          f"{args.distance_m:.0f} m em {'+X' if args.direction > 0 else '-X'}\n")
    print(f"{'alt (m)':>8} {'lateral':>8}  resultado")
    print("-" * 46)

    clear: list[tuple[float, float]] = []
    for lat_off in args.lateral_m:
        for alt in args.altitudes:
            ok, blocked = probe_corridor(
                client, conv, start_n, start_e + lat_off,
                args.distance_m, alt, args.direction, args.step_m,
            )
            if ok:
                print(f"{alt:8.0f} {lat_off:8.0f}  LIVRE")
                clear.append((alt, lat_off))
            else:
                print(f"{alt:8.0f} {lat_off:8.0f}  bloqueado em N={blocked:.0f}")

    print()
    if clear:
        alt, off = min(clear)  # menor altitude livre
        print(f"[OK] Menor altitude livre: {alt:.0f} m (lateral {off:+.0f} m)")
        print(f"     Use: --alt-m {alt:.0f}" + (f"  (com Y deslocado {off:+.0f} m)" if off else ""))
    else:
        print("[AVISO] Nenhuma combinacao testada ficou livre. "
              "Tente altitudes maiores, outro deslocamento lateral, ou trecho mais curto.")


if __name__ == "__main__":
    main()
