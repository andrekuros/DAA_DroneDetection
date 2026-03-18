"""
scenarios.py — Definições de cenários para experimentos de detecção de drones
=============================================================================
Cada ApproachScenario descreve:
  - Geometria de aproximação (distância, azimute, altitude relativa, velocidade)
  - Condições climáticas (chuva, neblina, poeira, neve)
  - Vento NED (North-East-Down) em m/s
  - Horário do dia para iluminação solar

Coordenadas NED do AirSim: X=Norte, Y=Leste, Z=para baixo (Z negativo = subindo).
"""

import math
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WeatherConfig:
    """Parâmetros climáticos. Valores de 0.0 (nenhum) a 1.0 (máximo)."""
    rain:  float = 0.0
    fog:   float = 0.0
    dust:  float = 0.0
    snow:  float = 0.0

    def is_clear(self) -> bool:
        return all(v == 0.0 for v in (self.rain, self.fog, self.dust, self.snow))

    def __str__(self) -> str:
        parts = []
        if self.rain  > 0: parts.append(f"Rain={self.rain:.1f}")
        if self.fog   > 0: parts.append(f"Fog={self.fog:.1f}")
        if self.dust  > 0: parts.append(f"Dust={self.dust:.1f}")
        if self.snow  > 0: parts.append(f"Snow={self.snow:.1f}")
        return ", ".join(parts) if parts else "Clear"


@dataclass
class ApproachScenario:
    """
    Descreve um único cenário de aproximação de drone intruso.

    Campos
    ------
    name              : identificador único do cenário
    distance_m        : distância inicial (m) entre o intruso e o observador
    azimuth_deg       : ângulo de chegada em graus (0=Norte/frente, 90=Leste/direita,
                        180=Sul/atrás, 270=Oeste/esquerda)
    altitude_offset_m : altitude do intruso relativa ao observador (+ = acima, - = abaixo)
    speed_ms          : velocidade de aproximação em m/s
    weather           : configuração climática
    wind_ned          : vento (N, E, D) em m/s – D positivo = vento para baixo
    time_of_day       : datetime string "YYYY-MM-DD HH:MM:SS" (afeta iluminação solar)
    description       : descrição legível do cenário
    """
    name:              str
    distance_m:        float
    azimuth_deg:       float
    altitude_offset_m: float
    speed_ms:          float
    weather:           WeatherConfig
    wind_ned:          tuple            = field(default_factory=lambda: (0.0, 0.0, 0.0))
    time_of_day:       str             = "2025-06-21 14:00:00"
    description:       str             = ""

    def start_position_ned(self, observer_ned: tuple = (0.0, 0.0, -5.0)) -> tuple:
        """
        Calcula a posição NED de início do drone intruso dado o azimute e distância.
        O intruso parte de `distance_m` metros do observador na direção `azimuth_deg`.

        Returns
        -------
        (x, y, z) em metros, sistema NED do AirSim.
        """
        az_rad = math.radians(self.azimuth_deg)
        dx = self.distance_m * math.cos(az_rad)   # Norte
        dy = self.distance_m * math.sin(az_rad)   # Leste
        # Z AirSim: negativo = acima do solo; altitude_offset é relativo ao observador
        dz = -self.altitude_offset_m              # positivo altitude_offset → Z menor (mais alto)

        ox, oy, oz = observer_ned
        return (ox + dx, oy + dy, oz + dz)

    def summary(self) -> str:
        az_dir = {0: "N", 45: "NE", 90: "E", 135: "SE",
                  180: "S", 225: "SW", 270: "W", 315: "NW"}.get(int(self.azimuth_deg), f"{self.azimuth_deg}°")
        wind_str = f"vento({self.wind_ned[0]:.0f},{self.wind_ned[1]:.0f})" \
                   if any(w != 0.0 for w in self.wind_ned) else "sem_vento"
        return (
            f"[{self.name}]  dist={self.distance_m:.0f}m  dir={az_dir}  "
            f"alt={self.altitude_offset_m:+.0f}m  vel={self.speed_ms:.0f}m/s  "
            f"clima={self.weather}  {wind_str}  hora={self.time_of_day[11:16]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Cenários pré-definidos
# ─────────────────────────────────────────────────────────────────────────────

# Helpers de horário
_DAWN  = "2025-06-21 06:00:00"
_DAY   = "2025-06-21 14:00:00"
_DUSK  = "2025-06-21 18:30:00"
_NIGHT = "2025-06-21 02:00:00"

# Helpers climáticos
_CLEAR     = WeatherConfig()
_LIGHT_RAIN = WeatherConfig(rain=0.3)
_HEAVY_RAIN = WeatherConfig(rain=1.0)
_FOG        = WeatherConfig(fog=0.6)
_DUST       = WeatherConfig(dust=0.5)
_SNOW       = WeatherConfig(snow=0.7)
_RAIN_FOG   = WeatherConfig(rain=0.5, fog=0.5)


SCENARIOS: list[ApproachScenario] = [

    # =====================================================================
    #  Camera: FOV=120deg, Yaw=0 (frente = Norte/+X)
    #  Angulos de aproximacao limitados a ±45deg do eixo frontal
    #  para garantir que o intruso aparece no campo de visao.
    # =====================================================================

    # ── Angulos de aproximacao (mesmo nivel, dia, 5 m/s) ──────────────────
    ApproachScenario(
        name="az000_day",
        distance_m=100, azimuth_deg=0, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="Frontal direto (0 deg), dia",
    ),
    ApproachScenario(
        name="az015_day",
        distance_m=100, azimuth_deg=15, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="15 deg direita, dia",
    ),
    ApproachScenario(
        name="az345_day",
        distance_m=100, azimuth_deg=345, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="15 deg esquerda, dia",
    ),
    ApproachScenario(
        name="az030_day",
        distance_m=100, azimuth_deg=30, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="30 deg direita, dia",
    ),
    ApproachScenario(
        name="az330_day",
        distance_m=100, azimuth_deg=330, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="30 deg esquerda, dia",
    ),
    ApproachScenario(
        name="az045_day",
        distance_m=100, azimuth_deg=45, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="45 deg direita (limite FOV), dia",
    ),
    ApproachScenario(
        name="az315_day",
        distance_m=100, azimuth_deg=315, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="45 deg esquerda (limite FOV), dia",
    ),

    # ── Altitude — frontal, dia ───────────────────────────────────────────
    ApproachScenario(
        name="az000_up10_day",
        distance_m=100, azimuth_deg=0, altitude_offset_m=10, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="Frontal, 10m acima, dia",
    ),
    ApproachScenario(
        name="az000_up25_day",
        distance_m=100, azimuth_deg=0, altitude_offset_m=25, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="Frontal, 25m acima (mergulho), dia",
    ),
    ApproachScenario(
        name="az000_dn10_day",
        distance_m=100, azimuth_deg=0, altitude_offset_m=-10, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="Frontal, 10m abaixo, dia",
    ),
    ApproachScenario(
        name="az000_dn25_day",
        distance_m=100, azimuth_deg=0, altitude_offset_m=-25, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="Frontal, 25m abaixo (subida), dia",
    ),

    # ── Angulo + altitude combinados ──────────────────────────────────────
    ApproachScenario(
        name="az030_up15_day",
        distance_m=100, azimuth_deg=30, altitude_offset_m=15, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="30 deg dir + 15m acima, dia",
    ),
    ApproachScenario(
        name="az330_dn15_day",
        distance_m=100, azimuth_deg=330, altitude_offset_m=-15, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="30 deg esq + 15m abaixo, dia",
    ),
    ApproachScenario(
        name="az045_up10_day",
        distance_m=100, azimuth_deg=45, altitude_offset_m=10, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="45 deg dir + 10m acima, dia",
    ),
    ApproachScenario(
        name="az315_dn10_day",
        distance_m=100, azimuth_deg=315, altitude_offset_m=-10, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="45 deg esq + 10m abaixo, dia",
    ),

    # ── Velocidade — frontal, dia ─────────────────────────────────────────
    ApproachScenario(
        name="az000_v03_day",
        distance_m=100, azimuth_deg=0, altitude_offset_m=0, speed_ms=3,
        weather=_CLEAR, time_of_day=_DAY,
        description="Frontal, 3 m/s (lento), dia",
    ),
    ApproachScenario(
        name="az000_v10_day",
        distance_m=100, azimuth_deg=0, altitude_offset_m=0, speed_ms=10,
        weather=_CLEAR, time_of_day=_DAY,
        description="Frontal, 10 m/s, dia",
    ),
    ApproachScenario(
        name="az000_v15_day",
        distance_m=100, azimuth_deg=0, altitude_offset_m=0, speed_ms=15,
        weather=_CLEAR, time_of_day=_DAY,
        description="Frontal, 15 m/s (rapido), dia",
    ),

    # ── Distancia — frontal, dia ──────────────────────────────────────────
    ApproachScenario(
        name="az000_50m_day",
        distance_m=50, azimuth_deg=0, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="Frontal, 50m (perto), dia",
    ),
    ApproachScenario(
        name="az000_200m_day",
        distance_m=200, azimuth_deg=0, altitude_offset_m=0, speed_ms=8,
        weather=_CLEAR, time_of_day=_DAY,
        description="Frontal, 200m (longe), dia",
    ),

    # ── Iluminacao — frontal, ceu limpo ───────────────────────────────────
    ApproachScenario(
        name="az000_dawn",
        distance_m=100, azimuth_deg=0, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAWN,
        description="Frontal, amanhecer (06h)",
    ),
    ApproachScenario(
        name="az000_dusk",
        distance_m=100, azimuth_deg=0, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, time_of_day=_DUSK,
        description="Frontal, entardecer (18h30)",
    ),
    ApproachScenario(
        name="az000_night",
        distance_m=100, azimuth_deg=0, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, time_of_day=_NIGHT,
        description="Frontal, noite (02h)",
    ),

    # ── Iluminacao + angulo ───────────────────────────────────────────────
    ApproachScenario(
        name="az030_dawn",
        distance_m=100, azimuth_deg=30, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAWN,
        description="30 deg dir, amanhecer",
    ),
    ApproachScenario(
        name="az330_dusk",
        distance_m=100, azimuth_deg=330, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, time_of_day=_DUSK,
        description="30 deg esq, entardecer",
    ),

    # ── Combinados realisticos ────────────────────────────────────────────
    ApproachScenario(
        name="az015_up10_v10_dawn",
        distance_m=150, azimuth_deg=15, altitude_offset_m=10, speed_ms=10,
        weather=_CLEAR, time_of_day=_DAWN,
        description="15 deg dir, 10m acima, 10m/s, 150m, amanhecer",
    ),
    ApproachScenario(
        name="az345_dn10_v10_dusk",
        distance_m=150, azimuth_deg=345, altitude_offset_m=-10, speed_ms=10,
        weather=_CLEAR, time_of_day=_DUSK,
        description="15 deg esq, 10m abaixo, 10m/s, 150m, entardecer",
    ),
    ApproachScenario(
        name="az030_up20_v15_day",
        distance_m=200, azimuth_deg=30, altitude_offset_m=20, speed_ms=15,
        weather=_CLEAR, time_of_day=_DAY,
        description="30 deg dir, 20m acima, 15m/s, 200m, dia",
    ),
    ApproachScenario(
        name="az000_up10_v03_night",
        distance_m=100, azimuth_deg=0, altitude_offset_m=10, speed_ms=3,
        weather=_CLEAR, time_of_day=_NIGHT,
        description="Frontal, 10m acima, 3m/s, noite (furtivo)",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Acesso por nome
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS_BY_NAME: dict[str, ApproachScenario] = {s.name: s for s in SCENARIOS}


def get_scenario(name: str) -> ApproachScenario:
    """Retorna cenário pelo nome ou lança KeyError com lista de opções."""
    if name not in SCENARIOS_BY_NAME:
        options = "\n  ".join(sorted(SCENARIOS_BY_NAME))
        raise KeyError(f"Cenário '{name}' não encontrado.\nOpções:\n  {options}")
    return SCENARIOS_BY_NAME[name]


def get_all_scenarios() -> list[ApproachScenario]:
    return list(SCENARIOS)


# ─────────────────────────────────────────────────────────────────────────────
# CLI rápido de listagem
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'#':>3}  {'Nome':<30} {'Dist':>6} {'Az':>5} {'Alt':>5} {'Vel':>5}  Clima           Hora")
    print("─" * 90)
    for i, s in enumerate(SCENARIOS, 1):
        print(
            f"{i:>3}  {s.name:<30} {s.distance_m:>5.0f}m {s.azimuth_deg:>4.0f}°"
            f" {s.altitude_offset_m:>+4.0f}m {s.speed_ms:>4.0f}m/s"
            f"  {str(s.weather):<15}  {s.time_of_day[11:16]}"
        )
    print(f"\nTotal: {len(SCENARIOS)} cenários")
