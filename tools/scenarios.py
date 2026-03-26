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
    rotate_observer_offset_deg : com --rotate-observer, soma-se ao bearing Observer→intruso
                        (0 = frente ao intruso). Valores ±30–50° mantêm o alvo dentro de FOV ~120°.
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
    rotate_observer_offset_deg: float = 0.0

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
_LIGHT_FOG  = WeatherConfig(fog=0.2)  # neblina leve (ex.: Block Dataset)
_DUST       = WeatherConfig(dust=0.5)
_SNOW       = WeatherConfig(snow=0.7)
_RAIN_FOG   = WeatherConfig(rain=0.5, fog=0.5)

# Suite exp30: neblina sempre <= 0.1 quando usada
_FOG_010 = WeatherConfig(fog=0.10)
_FOG_008 = WeatherConfig(fog=0.08)
_FOG_005 = WeatherConfig(fog=0.05)
_RAIN_FOG_SAFE = WeatherConfig(rain=0.40, fog=0.10)
_RAIN_MED = WeatherConfig(rain=0.55)
_RAIN_LIGHT = WeatherConfig(rain=0.28)
_DUST_MED = WeatherConfig(dust=0.40)
_DUST_LOW = WeatherConfig(dust=0.22)
_SNOW_MED = WeatherConfig(snow=0.50)
_SNOW_LOW = WeatherConfig(snow=0.28)


def _build_experiment30() -> list[ApproachScenario]:
    """
    30 cenarios: variacao de altitude_offset, azimute (bearing), distancia, velocidade,
    offset de yaw do Observer, clima (fog max 0.1), vento leve ocasional.
    Com --rotate-observer o bearing horizontal do intruso vem de azimuth_deg do cenario.
    """
    # (nome, dist, az, alt_off, v, weather, time_key, roff, wind)
    # time_key in dawn|day|dusk|night; wind None ou (n,e,d)
    _t = {"dawn": _DAWN, "day": _DAY, "dusk": _DUSK, "night": _NIGHT}
    specs: list[tuple] = [
        ("exp30_01_clr_day_fr", 90, 0, 0, 5, _CLEAR, "day", 0.0, None),
        ("exp30_02_f005_dusk_m15", 85, 0, -15, 5, _FOG_005, "dusk", 15.0, None),
        ("exp30_03_f010_dawn_up8", 95, 15, 8, 6, _FOG_010, "dawn", -20.0, None),
        ("exp30_04_rain_day_az25", 100, 25, 12, 7, _RAIN_LIGHT, "day", 0.0, (2.0, 0.5, 0.0)),
        ("exp30_05_clr_dusk_az340_dn8", 110, 340, -8, 5, _CLEAR, "dusk", -35.0, None),
        ("exp30_06_dust_day_az45", 75, 45, 0, 8, _DUST_LOW, "day", 10.0, None),
        ("exp30_07_snow_dawn_az330", 88, 330, 18, 4, _SNOW_LOW, "dawn", 25.0, None),
        ("exp30_08_clr_day_az35_up20", 115, 35, 20, 6, _CLEAR, "day", -12.0, None),
        ("exp30_09_rf_dusk_up22", 92, 0, 22, 5, _RAIN_FOG_SAFE, "dusk", 0.0, None),
        ("exp30_10_clr_night_up10", 100, 0, 10, 3, _CLEAR, "night", 0.0, None),
        ("exp30_11_f008_day_az315_dn18", 82, 315, -18, 7, _FOG_008, "day", 40.0, None),
        ("exp30_12_clr_day_az20", 72, 20, 6, 8, _CLEAR, "day", 0.0, None),
        ("exp30_13_rain_day_dn12", 105, 10, -12, 6, _RAIN_MED, "day", -25.0, None),
        ("exp30_14_dust_dusk_az30", 98, 30, -5, 5, _DUST_MED, "dusk", 0.0, (0.0, 3.0, 0.0)),
        ("exp30_15_f005_clrish_dawn", 80, 345, 0, 5, _FOG_005, "dawn", -10.0, None),
        ("exp30_16_clr_day_long120", 120, 0, 15, 7, _CLEAR, "day", 20.0, None),
        ("exp30_17_snow_day_az15", 94, 15, -10, 5, _SNOW_MED, "day", 0.0, None),
        ("exp30_18_f010_day_az325_up25", 88, 325, 25, 6, _FOG_010, "day", -30.0, None),
        ("exp30_19_clr_dusk_fast8", 70, 0, 0, 8, _CLEAR, "dusk", 0.0, None),
        ("exp30_20_rain_dawn_slow4", 100, 335, 14, 4, _RAIN_LIGHT, "dawn", 18.0, None),
        ("exp30_21_dust_day_az40", 90, 40, 8, 6, _DUST_MED, "day", -8.0, None),
        ("exp30_22_clr_day_az350_dn20", 102, 350, -20, 5, _CLEAR, "day", 0.0, None),
        ("exp30_23_f008_rainish_dusk", 86, 5, 0, 6, WeatherConfig(rain=0.35, fog=0.08), "dusk", 12.0, None),
        ("exp30_24_snow_dusk_az355", 91, 355, 22, 5, _SNOW_LOW, "dusk", -22.0, None),
        ("exp30_25_clr_day_az50", 78, 50, 4, 7, _CLEAR, "day", 0.0, None),
        ("exp30_26_clr_dawn_az310_up16", 96, 310, 16, 5, _CLEAR, "dawn", 30.0, None),
        ("exp30_27_f005_night_az0", 85, 0, -6, 4, _FOG_005, "night", 0.0, None),
        ("exp30_28_rain_day_az28", 108, 28, -14, 7, _RAIN_MED, "day", -15.0, (1.5, -1.0, 0.0)),
        ("exp30_29_dust_dawn", 93, 8, 10, 5, _DUST_LOW, "dawn", 0.0, None),
        ("exp30_30_rf_day_mixed", 99, 22, -16, 6, _RAIN_FOG_SAFE, "day", 8.0, None),
    ]
    out: list[ApproachScenario] = []
    for name, dist, az, alt, v, wx, tk, roff, wind in specs:
        wn = wind if wind is not None else (0.0, 0.0, 0.0)
        out.append(
            ApproachScenario(
                name=name,
                distance_m=float(dist),
                azimuth_deg=float(az),
                altitude_offset_m=float(alt),
                speed_ms=float(v),
                weather=wx,
                wind_ned=wn,
                time_of_day=_t[tk],
                rotate_observer_offset_deg=float(roff),
                description=f"Exp30: alt={alt:+.0f}m az={az:.0f}° v={v}m/s roff={roff:+.0f}° {wx} {_t[tk][11:16]}",
            )
        )
    return out


EXPERIMENT30_SCENARIOS: list[ApproachScenario] = _build_experiment30()
EXPERIMENT30_SCENARIO_NAMES: tuple[str, ...] = tuple(s.name for s in EXPERIMENT30_SCENARIOS)


SCENARIOS: list[ApproachScenario] = [

    # =====================================================================
    #  Camera: FOV=120deg, Yaw=0 (frente = Norte/+X)
    #  Angulos de aproximacao limitados a ±45deg do eixo frontal
    #  para garantir que o intruso aparece no campo de visao.
    #  Com --rotate-observer: bearing horizontal do intruso = azimuth_deg do cenario;
    #  yaw do Observer = bearing(Observer→intruso) + rotate_observer_offset_deg.
    #  Use --approach-lateral-offset (padrao 4 m com --rotate-observer) + separacao vertical
    #  para evitar colisao fisica com o veiculo observador.
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

    # ── Teste rápido: clima + hora (projeção vs segmentação) ───────────────
    ApproachScenario(
        name="labeltest_clear_day",
        distance_m=80, azimuth_deg=0, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        description="Teste labels: céu limpo, dia",
    ),
    ApproachScenario(
        name="labeltest_fog_dusk",
        distance_m=80, azimuth_deg=0, altitude_offset_m=0, speed_ms=5,
        weather=_FOG, time_of_day=_DUSK,
        description="Teste labels: neblina + entardecer",
    ),
    ApproachScenario(
        name="labeltest_rain_dawn",
        distance_m=80, azimuth_deg=0, altitude_offset_m=0, speed_ms=5,
        weather=_LIGHT_RAIN, time_of_day=_DAWN,
        description="Teste labels: chuva leve + amanhecer",
    ),

    # ── Block Dataset: --rotate-observer + bearing automático; offset só desvia a vista (FOV 120°).
    ApproachScenario(
        name="block_rot_clear_day",
        distance_m=80, azimuth_deg=0, altitude_offset_m=0, speed_ms=5,
        weather=_CLEAR, time_of_day=_DAY,
        rotate_observer_offset_deg=0.0,
        description="Block: frente ao intruso, limpo, dia",
    ),
    ApproachScenario(
        name="block_rot_fog_dusk",
        distance_m=80, azimuth_deg=0, altitude_offset_m=0, speed_ms=5,
        weather=_LIGHT_FOG, time_of_day=_DUSK,
        rotate_observer_offset_deg=45.0,
        description="Block: +45° no yaw, neblina leve, entardecer",
    ),
    ApproachScenario(
        name="block_rot_rain_dawn",
        distance_m=80, azimuth_deg=0, altitude_offset_m=0, speed_ms=5,
        weather=_LIGHT_RAIN, time_of_day=_DAWN,
        rotate_observer_offset_deg=-45.0,
        description="Block: -45° no yaw, chuva leve, amanhecer",
    ),
] + EXPERIMENT30_SCENARIOS


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
