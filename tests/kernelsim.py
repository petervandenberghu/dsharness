"""Draait code in een echte kernel en bootst klikken vanuit de frontend na."""
from __future__ import annotations

import queue
import time

from jupyter_client.manager import start_new_kernel


class KernelSim:
    def __init__(self):
        self.km, self.kc = start_new_kernel(kernel_name="python3")
        self.knoppen: dict[str, str] = {}  # description -> comm_id (laatste)
        self.textareas: list[str] = []
        self.stdout: list[str] = []
        self.errors: list[str] = []

    def stop(self):
        self.kc.stop_channels()
        self.km.shutdown_kernel(now=True)

    def _verwerk(self, msg) -> None:
        t, c = msg["msg_type"], msg["content"]
        if t == "comm_open":
            state = c.get("data", {}).get("state", {})
            if state.get("_model_name") == "ButtonModel":
                self.knoppen[state.get("description")] = c["comm_id"]
            if state.get("_model_name") == "TextareaModel":
                self.textareas.append(c["comm_id"])
        elif t == "stream":
            self.stdout.append(c["text"])
        elif t == "error":
            self.errors.append("\n".join(c["traceback"]))

    def pomp(self, tot: float) -> bool:
        """Lees iopub tot `tot` (monotonic). Geeft True als de kernel idle werd."""
        idle = False
        while time.monotonic() < tot:
            try:
                msg = self.kc.get_iopub_msg(timeout=0.1)
            except queue.Empty:
                continue
            self._verwerk(msg)
            if msg["msg_type"] == "status" and msg["content"]["execution_state"] == "idle":
                if msg["parent_header"].get("msg_id") == self._laatste:
                    idle = True
                    break
        return idle

    def voer_uit(self, code: str, wacht: float = 15) -> bool:
        self._laatste = self.kc.execute(code)
        return self.pomp(time.monotonic() + wacht)

    def start(self, code: str) -> None:
        """Start code zonder te wachten tot de cel klaar is."""
        self._laatste = self.kc.execute(code)

    def wacht_op_knop(self, naam: str, wacht: float = 15) -> str:
        eind = time.monotonic() + wacht
        while naam not in self.knoppen and time.monotonic() < eind:
            try:
                self._verwerk(self.kc.get_iopub_msg(timeout=0.1))
            except queue.Empty:
                pass
        if naam not in self.knoppen:
            raise TimeoutError(f"Knop {naam!r} niet verschenen")
        return self.knoppen[naam]

    def leeg(self, duur: float = 0.4) -> None:
        """Verwerk openstaande iopub-berichten, zodat nieuwe widgets bekend zijn."""
        eind = time.monotonic() + duur
        while time.monotonic() < eind:
            try:
                self._verwerk(self.kc.get_iopub_msg(timeout=0.05))
            except queue.Empty:
                pass

    def _comm(self, comm_id: str, data: dict) -> None:
        self.kc.shell_channel.send(self.kc.session.msg("comm_msg", {"comm_id": comm_id, "data": data}))

    def klik(self, naam: str) -> None:
        self.leeg()
        self._comm(self.wacht_op_knop(naam), {"method": "custom", "content": {"event": "click"}})

    def typ(self, tekst: str, van_achter: int = 1) -> None:
        """Typ in het laatste tekstveld (of het n-de van achteren)."""
        self.leeg()
        self._comm(self.textareas[-van_achter], {"method": "update", "state": {"value": tekst}, "buffer_paths": []})
