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

    # ── Geometria de aproximação — condições ideais ───────────────────────────
    ApproachScenario(
        name="frontal_clear_day",
        distance_m=200, azimuth_deg=0, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="Aproximação frontal direta, 200m, dia claro",
    ),
    ApproachScenario(
        name="lateral_right_clear_day",
        distance_m=200, azimuth_deg=90, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="Aproximação pelo lado direito (Leste), 200m",
    ),
    ApproachScenario(
        name="lateral_left_clear_day",
        distance_m=200, azimuth_deg=270, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="Aproximação pelo lado esquerdo (Oeste), 200m",
    ),
    ApproachScenario(
        name="diagonal_NE_clear_day",
        distance_m=200, azimuth_deg=45, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="Aproximação diagonal NE, 200m",
    ),
    ApproachScenario(
        name="rear_clear_day",
        distance_m=200, azimuth_deg=180, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="Aproximação por trás (Sul), 200m",
    ),

    # ── Variação de altitude ──────────────────────────────────────────────────
    ApproachScenario(
        name="frontal_high_altitude",
        distance_m=200, azimuth_deg=0, altitude_offset_m=20, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="Intruso 20m acima do observador — ângulo de mergulho",
    ),
    ApproachScenario(
        name="frontal_low_altitude",
        distance_m=200, azimuth_deg=0, altitude_offset_m=-10, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="Intruso 10m abaixo do observador — ângulo de subida",
    ),
    ApproachScenario(
        name="frontal_extreme_high",
        distance_m=200, azimuth_deg=0, altitude_offset_m=50, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="Intrusão a 50m acima — dive vertical pronunciado",
    ),

    # ── Variação de distância ─────────────────────────────────────────────────
    ApproachScenario(
        name="close_range_50m",
        distance_m=50, azimuth_deg=0, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="Aproximação curta — 50m (drone já está próximo)",
    ),
    ApproachScenario(
        name="medium_range_100m",
        distance_m=100, azimuth_deg=0, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="Alcance médio — 100m",
    ),
    ApproachScenario(
        name="long_range_400m",
        distance_m=400, azimuth_deg=0, altitude_offset_m=0, speed_ms=10,
        weather=_CLEAR, time_of_day=_DAY,
        description="Longo alcance — 400m, velocidade maior",
    ),

    # ── Variação de velocidade ────────────────────────────────────────────────
    ApproachScenario(
        name="very_slow_2ms",
        distance_m=200, azimuth_deg=0, altitude_offset_m=0, speed_ms=2,
        weather=_CLEAR, time_of_day=_DAY,
        description="Aproximação lenta: drone furtivo a 2 m/s",
    ),
    ApproachScenario(
        name="fast_10ms",
        distance_m=200, azimuth_deg=0, altitude_offset_m=0, speed_ms=10,
        weather=_CLEAR, time_of_day=_DAY,
        description="Aproximação rápida a 10 m/s",
    ),
    ApproachScenario(
        name="sprint_20ms",
        distance_m=200, azimuth_deg=0, altitude_offset_m=0, speed_ms=20,
        weather=_CLEAR, time_of_day=_DAY,
        description="Sprint a 20 m/s — máxima velocidade",
    ),

    # ── Variação de horário (iluminação) ──────────────────────────────────────
    ApproachScenario(
        name="frontal_clear_dawn",
        distance_m=200, azimuth_deg=0, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAWN,
        description="Amanhecer — iluminação lateral baixa",
    ),
    ApproachScenario(
        name="frontal_clear_dusk",
        distance_m=200, azimuth_deg=0, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, time_of_day=_DUSK,
        description="Entardecer — drone contra-luz",
    ),
    ApproachScenario(
        name="frontal_clear_night",
        distance_m=200, azimuth_deg=0, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, time_of_day=_NIGHT,
        description="Noite — sem iluminação natural",
    ),

    # ── Condições climáticas adversas ────────────────────────────────────────
    ApproachScenario(
        name="frontal_light_rain",
        distance_m=200, azimuth_deg=0, altitude_offset_m=0, speed_ms=5,
        weather=_LIGHT_RAIN, time_of_day=_DAY,
        description="Chuva leve (Rain=0.3)",
    ),
    ApproachScenario(
        name="frontal_heavy_rain",
        distance_m=200, azimuth_deg=0, altitude_offset_m=0, speed_ms=5,
        weather=_HEAVY_RAIN, time_of_day=_DAY,
        description="Chuva forte (Rain=1.0) — visibilidade muito reduzida",
    ),
    ApproachScenario(
        name="frontal_fog",
        distance_m=200, azimuth_deg=0, altitude_offset_m=0, speed_ms=5,
        weather=_FOG, time_of_day=_DAY,
        description="Neblina densa (Fog=0.6)",
    ),
    ApproachScenario(
        name="frontal_dust",
        distance_m=200, azimuth_deg=0, altitude_offset_m=0, speed_ms=5,
        weather=_DUST, time_of_day=_DAY,
        description="Tempestade de poeira (Dust=0.5)",
    ),
    ApproachScenario(
        name="frontal_snow",
        distance_m=200, azimuth_deg=0, altitude_offset_m=0, speed_ms=5,
        weather=_SNOW, time_of_day=_DAY,
        description="Neve moderada (Snow=0.7)",
    ),

    # ── Vento ─────────────────────────────────────────────────────────────────
    ApproachScenario(
        name="frontal_crosswind_E10",
        distance_m=200, azimuth_deg=0, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, wind_ned=(0.0, 10.0, 0.0), time_of_day=_DAY,
        description="Vento cruzado leste a 10 m/s — deriva lateral",
    ),
    ApproachScenario(
        name="frontal_headwind_N15",
        distance_m=200, azimuth_deg=0, altitude_offset_m=0, speed_ms=8,
        weather=_CLEAR, wind_ned=(15.0, 0.0, 0.0), time_of_day=_DAY,
        description="Vento de cauda norte 15 m/s (empurra o intruso)",
    ),

    # ── Cenários combinados / worst-case ─────────────────────────────────────
    ApproachScenario(
        name="rainy_night",
        distance_m=200, azimuth_deg=0, altitude_offset_m=0, speed_ms=5,
        weather=_LIGHT_RAIN, time_of_day=_NIGHT,
        description="Chuva leve à noite — condição difícil combinada",
    ),
    ApproachScenario(
        name="worst_case",
        distance_m=100, azimuth_deg=45, altitude_offset_m=15, speed_ms=15,
        weather=_RAIN_FOG, wind_ned=(5.0, 8.0, 0.0),
        time_of_day=_NIGHT,
        description="Pior caso: curto alcance, diagonal, alta velocidade, chuva+neblina, noite",
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
