"""
Screen Capture - Captura em Tempo Real para Simulador (DJI/Colosseum)
=====================================================================
Usa mss para captura de alta velocidade e OpenCV para exibição do feed.
Suporta seleção de janela por título (Windows).

Dependências:
    pip install mss opencv-python pywin32

Uso:
    python screen_capture.py                    # Lista janelas disponíveis e inicia
    python screen_capture.py --title "AirSim"   # Seleciona janela pelo título diretamente
    python screen_capture.py --region 0 0 1280 720  # Captura região específica (x, y, w, h)
    python screen_capture.py --output feed.avi  # Grava o feed em arquivo
"""

import sys
import time
import argparse
import threading
from collections import deque

import cv2
import mss
import mss.tools
import numpy as np

# Tenta importar win32gui (Windows); em outros SOs, ignora funcionalidade de janela
try:
    import win32gui
    import win32con
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False


# ─────────────────────────────────────────────────────────────────────────────
# Utilitários de janela (Windows)
# ─────────────────────────────────────────────────────────────────────────────

def list_windows() -> list[tuple[int, str]]:
    """Retorna lista de (hwnd, título) de janelas visíveis."""
    if not HAS_WIN32:
        print("[AVISO] pywin32 não instalado – listagem de janelas indisponível.")
        return []

    results = []

    def _enum_cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                results.append((hwnd, title))

    win32gui.EnumWindows(_enum_cb, None)
    return sorted(results, key=lambda x: x[1].lower())


def get_window_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """Retorna (x, y, largura, altura) da janela ou None se inválida."""
    if not HAS_WIN32:
        return None
    try:
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        return left, top, right - left, bottom - top
    except Exception:
        return None


def bring_window_to_front(hwnd: int):
    """Coloca a janela em primeiro plano."""
    if HAS_WIN32:
        try:
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            pass


def find_window_by_title(title_substring: str) -> tuple[int | None, str]:
    """Encontra o HWND de uma janela cujo título contenha title_substring."""
    windows = list_windows()
    matches = [(h, t) for h, t in windows if title_substring.lower() in t.lower()]
    if not matches:
        return None, ""
    if len(matches) > 1:
        print(f"[INFO] Múltiplas janelas encontradas para '{title_substring}':")
        for i, (h, t) in enumerate(matches):
            print(f"  [{i}] {t}")
        choice = input("Selecione o número da janela: ").strip()
        idx = int(choice) if choice.isdigit() else 0
        return matches[min(idx, len(matches) - 1)]
    return matches[0]


def interactive_window_select() -> tuple[int | None, str, tuple | None]:
    """Menu interativo para selecionar janela. Retorna (hwnd, título, rect)."""
    windows = list_windows()
    if not windows:
        print("[ERRO] Nenhuma janela encontrada.")
        return None, "", None

    print("\n┌─── Janelas Disponíveis ─────────────────────────────────────────")
    for i, (hwnd, title) in enumerate(windows):
        print(f"│  [{i:3d}] {title}")
    print("└────────────────────────────────────────────────────────────────")
    print("  [  R] Capturar região manual")
    print("  [  Q] Sair")

    choice = input("\nSelecione uma janela (número ou letra): ").strip().upper()

    if choice == "Q":
        sys.exit(0)
    if choice == "R":
        return None, "Região Manual", None  # Sinaliza captura manual

    try:
        idx = int(choice)
        hwnd, title = windows[idx]
        rect = get_window_rect(hwnd)
        print(f"[OK] Janela selecionada: '{title}' | Rect: {rect}")
        return hwnd, title, rect
    except (ValueError, IndexError):
        print("[ERRO] Seleção inválida.")
        return None, "", None


# ─────────────────────────────────────────────────────────────────────────────
# Captura principal
# ─────────────────────────────────────────────────────────────────────────────

class ScreenCapture:
    """
    Captura de tela de alta performance com mss.

    Parâmetros
    ----------
    monitor : dict
        Dicionário no formato {"left": x, "top": y, "width": w, "height": h}.
    target_fps : int
        FPS alvo (o loop tentará manter este valor).
    scale : float
        Fator de escala da imagem exibida (1.0 = tamanho real).
    show_fps : bool
        Se True, sobrepõe o FPS atual no frame exibido.
    output_path : str | None
        Caminho para gravar o vídeo de saída (None = não grava).
    hwnd : int | None
        Handle da janela alvo para atualizar rect dinamicamente.
    """

    def __init__(
        self,
        monitor: dict,
        target_fps: int = 60,
        scale: float = 1.0,
        show_fps: bool = True,
        output_path: str | None = None,
        hwnd: int | None = None,
    ):
        self.monitor = monitor
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.scale = scale
        self.show_fps = show_fps
        self.output_path = output_path
        self.hwnd = hwnd

        self._running = False
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._fps_window = deque(maxlen=30)  # últimos 30 timestamps de frame

        self._writer: cv2.VideoWriter | None = None

    # ── Configuração de gravação ──────────────────────────────────────────────

    def _init_writer(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self._writer = cv2.VideoWriter(self.output_path, fourcc, self.target_fps, (w, h))

    # ── Loop de captura (thread separada) ────────────────────────────────────

    def _capture_loop(self):
        """Thread de captura dedicada – mantém _latest_frame atualizado."""
        with mss.mss() as sct:
            while self._running:
                t0 = time.perf_counter()

                # Atualiza rect da janela dinamicamente se hwnd disponível
                if self.hwnd and HAS_WIN32:
                    rect = get_window_rect(self.hwnd)
                    if rect:
                        x, y, w, h = rect
                        self.monitor = {"left": x, "top": y, "width": max(w, 1), "height": max(h, 1)}

                # Captura
                try:
                    img = sct.grab(self.monitor)
                except Exception as e:
                    print(f"[ERRO captura] {e}")
                    time.sleep(0.1)
                    continue

                # Converte BGRA → BGR (OpenCV)
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                if self.scale != 1.0:
                    new_w = max(1, int(frame.shape[1] * self.scale))
                    new_h = max(1, int(frame.shape[0] * self.scale))
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                with self._frame_lock:
                    self._latest_frame = frame

                # Controle de taxa
                elapsed = time.perf_counter() - t0
                sleep = self.frame_time - elapsed
                if sleep > 0:
                    time.sleep(sleep)

    # ── Loop de exibição (thread principal) ──────────────────────────────────

    def run(self, window_title: str = "Screen Capture"):
        """Inicia captura e exibe o feed na janela OpenCV."""
        self._running = True
        cap_thread = threading.Thread(target=self._capture_loop, daemon=True)
        cap_thread.start()

        print("\n[CAPTURA INICIADA]")
        print("  Pressione  Q  ou  ESC  para encerrar")
        print("  Pressione  S  para salvar frame atual como PNG")
        print("  Pressione  P  para pausar/retomar")
        print("  Pressione  +/-  para ajustar escala\n")

        paused = False
        frame_count = 0
        save_count = 0

        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

        while True:
            t_display = time.perf_counter()

            with self._frame_lock:
                frame = self._latest_frame.copy() if self._latest_frame is not None else None

            if frame is not None and not paused:
                # Calcula FPS
                self._fps_window.append(t_display)
                if len(self._fps_window) >= 2:
                    fps = (len(self._fps_window) - 1) / (
                        self._fps_window[-1] - self._fps_window[0]
                    )
                else:
                    fps = 0.0

                display = frame.copy()

                if self.show_fps:
                    label = f"FPS: {fps:.1f}  |  {self.monitor['width']}x{self.monitor['height']}"
                    cv2.rectangle(display, (0, 0), (350, 28), (0, 0, 0), -1)
                    cv2.putText(
                        display, label, (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 80), 1, cv2.LINE_AA,
                    )

                cv2.imshow(window_title, display)

                # Gravação em arquivo
                if self._writer is None and self.output_path:
                    self._init_writer(frame)
                if self._writer:
                    self._writer.write(frame)

                frame_count += 1
            elif paused:
                # Exibe frame congelado com aviso
                if frame is not None:
                    paused_frame = frame.copy()
                    cv2.rectangle(paused_frame, (0, 0), (110, 28), (0, 0, 0), -1)
                    cv2.putText(
                        paused_frame, "PAUSADO", (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 1, cv2.LINE_AA,
                    )
                    cv2.imshow(window_title, paused_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # Q ou ESC
                break
            elif key in (ord("s"), ord("S")):
                if frame is not None:
                    fn = f"capture_{save_count:04d}.png"
                    cv2.imwrite(fn, frame)
                    print(f"[SALVO] {fn}")
                    save_count += 1
            elif key in (ord("p"), ord("P")):
                paused = not paused
                print("[PAUSADO]" if paused else "[RETOMADO]")
            elif key == ord("+"):
                self.scale = min(self.scale + 0.1, 3.0)
                print(f"[ESCALA] {self.scale:.1f}x")
            elif key == ord("-"):
                self.scale = max(self.scale - 0.1, 0.1)
                print(f"[ESCALA] {self.scale:.1f}x")

        # ── Encerramento ──────────────────────────────────────────────────────
        self._running = False
        cap_thread.join(timeout=2)
        if self._writer:
            self._writer.release()
        cv2.destroyAllWindows()
        print(f"\n[ENCERRADO] Total de frames exibidos: {frame_count}")


# ─────────────────────────────────────────────────────────────────────────────
# Ponto de entrada
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Captura em tempo real de janela/região com mss + OpenCV."
    )
    p.add_argument(
        "--title", "-t", type=str, default=None,
        help="Substring do título da janela a capturar (ex: 'AirSim', 'DJI').",
    )
    p.add_argument(
        "--region", "-r", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
        default=None,
        help="Região manual: X Y Largura Altura (em pixels, relativo à tela).",
    )
    p.add_argument(
        "--fps", type=int, default=60,
        help="FPS alvo (padrão: 60).",
    )
    p.add_argument(
        "--scale", type=float, default=1.0,
        help="Fator de escala da imagem exibida (padrão: 1.0).",
    )
    p.add_argument(
        "--no-fps", action="store_true",
        help="Oculta o contador de FPS na tela.",
    )
    p.add_argument(
        "--output", "-o", type=str, default=None,
        help="Caminho do arquivo de vídeo de saída (ex: feed.avi).",
    )
    p.add_argument(
        "--list", "-l", action="store_true",
        help="Lista todas as janelas visíveis e encerra.",
    )
    p.add_argument(
        "--monitor", "-m", type=int, default=1,
        help="Índice do monitor para captura de tela cheia (padrão: 1 = primário).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── Apenas listar janelas ─────────────────────────────────────────────────
    if args.list:
        windows = list_windows()
        if windows:
            print(f"\n{'ID':>6}  Título")
            print("─" * 60)
            for hwnd, title in windows:
                print(f"{hwnd:>6}  {title}")
        else:
            print("Nenhuma janela encontrada (verifique se pywin32 está instalado).")
        return

    # ── Determina região/monitor de captura ──────────────────────────────────
    hwnd = None
    window_title_display = "Screen Capture"

    if args.region:
        x, y, w, h = args.region
        monitor = {"left": x, "top": y, "width": w, "height": h}
        print(f"[REGIÃO] Capturando: x={x} y={y} w={w} h={h}")

    elif args.title:
        hwnd, found_title = find_window_by_title(args.title)
        if hwnd is None:
            print(f"[ERRO] Janela com título '{args.title}' não encontrada.")
            sys.exit(1)
        rect = get_window_rect(hwnd)
        if rect is None:
            print("[ERRO] Não foi possível obter o rect da janela.")
            sys.exit(1)
        x, y, w, h = rect
        monitor = {"left": x, "top": y, "width": max(w, 1), "height": max(h, 1)}
        window_title_display = found_title
        print(f"[JANELA] '{found_title}' | Rect: {rect}")

    else:
        # Menu interativo
        result = interactive_window_select()
        if result is None or result[0] is None and result[2] is None:
            # Captura de tela cheia
            with mss.mss() as sct:
                mon = sct.monitors[args.monitor]
            monitor = {
                "left": mon["left"], "top": mon["top"],
                "width": mon["width"], "height": mon["height"],
            }
            print(f"[TELA CHEIA] Monitor {args.monitor}: {monitor}")
        else:
            hwnd, window_title_display, rect = result
            if rect:
                x, y, w, h = rect
                monitor = {"left": x, "top": y, "width": max(w, 1), "height": max(h, 1)}
            else:
                # Região manual
                print("\nDigite a região manualmente:")
                x   = int(input("  X (esquerda): "))
                y   = int(input("  Y (topo):     "))
                w   = int(input("  Largura:      "))
                h   = int(input("  Altura:       "))
                monitor = {"left": x, "top": y, "width": w, "height": h}

    # ── Inicia captura ────────────────────────────────────────────────────────
    if hwnd and HAS_WIN32:
        bring_window_to_front(hwnd)
        time.sleep(0.3)  # Aguarda janela ficar em frente

    capture = ScreenCapture(
        monitor=monitor,
        target_fps=args.fps,
        scale=args.scale,
        show_fps=not args.no_fps,
        output_path=args.output,
        hwnd=hwnd,
    )
    capture.run(window_title=f"Capture – {window_title_display}")


if __name__ == "__main__":
    main()
