"""Cockpit: voorstelkaarten en PO-vragen als ipywidgets in een notebook.

Een workflow wacht op de PO met `await cockpit.vraag_akkoord(...)` en draait als
achtergrondtaak via `cockpit.start(...)`. Zo komen klikken binnen terwijl de taak wacht,
in ipykernel 6 en 7. (Direct `await` in een cel werkt alleen in ipykernel 7; zie scripts/check_async.py.)
"""
from __future__ import annotations

import asyncio
import html
import io
import traceback
from typing import Any, Callable, Coroutine

import ipywidgets as w

from .beslislog import Beslislog
from .contracts import Besluit, Keuze, Voorstel

KLEUR = {"goed": "#2C7A57", "twijfel": "#A86300", "fout": "#B42318"}
LABEL = {"goed": "Reviewer: goed", "twijfel": "Reviewer: twijfel", "fout": "Reviewer: fout"}

CSS = """
<style>
.dsh-kaart{border:1px solid var(--jp-border-color1,#D6DBE2);border-left:5px solid var(--dsh-k);border-radius:10px;
  padding:12px 16px;background:var(--jp-layout-color1,#fff);color:var(--jp-ui-font-color1,#172033);font-size:14px;line-height:1.5;max-width:820px}
.dsh-kaart h3{margin:0 0 8px;font-size:16px}
.dsh-kaart dl{display:grid;grid-template-columns:120px minmax(0,1fr);gap:4px 12px;margin:0}
.dsh-kaart dt{color:var(--jp-ui-font-color2,#5B6577);font-weight:600}
.dsh-kaart dd{margin:0}
.dsh-badge{display:inline-block;padding:1px 9px;border-radius:999px;font-size:12px;font-weight:700;color:#fff;background:var(--dsh-k);margin-bottom:6px}
.dsh-review{margin-top:10px;padding:8px 10px;border-radius:8px;background:var(--jp-layout-color2,#F4F6F8)}
.dsh-meta{color:var(--jp-ui-font-color2,#5B6577);font-size:12px}
</style>
"""


def _bronnen_html(bronnen) -> str:
    return "; ".join(f'<a href="{html.escape(b.url)}" target="_blank">{html.escape(b.titel)}</a>' for b in bronnen)


def _kaart_html(v: Voorstel) -> str:
    e = html.escape
    rij = lambda k, val: f"<dt>{k}</dt><dd>{val}</dd>" if val else ""
    r = v.review
    return CSS + f"""
<div class="dsh-kaart" style="--dsh-k:{KLEUR[r.oordeel]}">
  <div class="dsh-meta">{e(v.fase)} · voorstel {e(v.id)}{' · ' + e(', '.join(v.kolommen)) if v.kolommen else ''}</div>
  <h3>{e(v.actie)}</h3>
  <dl>
    {rij('Observatie', e(v.observatie))}
    {rij('Onderzocht', e(v.onderzocht))}
    {rij('PO-kennis', e(v.po_kennis))}
    {rij('Plan', e(v.plan))}
    {rij('Waarom', e(v.waarom))}
    {rij('Bron', _bronnen_html(v.bronnen))}
    {rij('Alternatief', e(v.alternatief or ''))}
  </dl>
  <div class="dsh-review"><span class="dsh-badge">{LABEL[r.oordeel]}</span><div>{e(r.toelichting)}</div>
    {'<div class="dsh-meta">' + _bronnen_html(r.bronnen) + '</div>' if r.bronnen else ''}</div>
  <p style="margin:10px 0 0"><b>Is dit akkoord?</b></p>
</div>"""


def _figuur_widget(figuur: Any) -> w.Widget | None:
    if figuur is None:
        return None
    buf = io.BytesIO()
    figuur.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return w.Image(value=buf.getvalue(), format="png", layout=w.Layout(max_width="560px"))


class _Kaart:
    """Eén voorstelkaart. Roept `klaar(besluit)` aan zodra de PO besloten heeft."""

    def __init__(self, v: Voorstel, klaar: Callable[[Besluit], None], figuur: Any = None):
        self.v, self.klaar = v, klaar
        fout = v.review.oordeel == "fout"
        self.knoppen = {
            "akkoord": w.Button(description="Akkoord", button_style="success", disabled=fout,
                                tooltip="Niet mogelijk bij een 'fout' van de Reviewer" if fout else ""),
            "aanpassen": w.Button(description="Aanpassen"),
            "afwijzen": w.Button(description="Afwijzen", button_style="danger"),
        }
        if fout:
            self.knoppen["overrulen"] = w.Button(description="Overrulen", button_style="warning")
        self.motivatie = w.Textarea(placeholder="Motivatie", layout=w.Layout(width="560px", height="70px"))
        self.bevestig = w.Button(description="Bevestigen", button_style="primary")
        self.motivatie_box = w.VBox([self.motivatie, self.bevestig], layout=w.Layout(display="none"))
        self.status = w.HTML()
        self.gekozen: Keuze | None = None
        self.besloten = False  # na een besluit negeert de kaart alle verdere events
        for keuze, knop in self.knoppen.items():
            knop.on_click(lambda _b, k=keuze: self._kies(k))
        self.bevestig.on_click(lambda _b: self._bevestig())
        delen = [w.HTML(_kaart_html(v))]
        if (fig := _figuur_widget(figuur)) is not None:
            delen.append(fig)
        delen += [w.HBox(list(self.knoppen.values())), self.motivatie_box, self.status]
        self.widget = w.VBox(delen, layout=w.Layout(margin="0 0 16px"))

    def _kies(self, keuze: Keuze) -> None:
        if self.besloten:
            return
        self.gekozen = keuze
        if keuze == "akkoord":
            self._rond_af("")
            return
        verplicht = keuze in ("aanpassen", "overrulen")
        self.motivatie.placeholder = f"Motivatie bij '{keuze}'" + (" (verplicht)" if verplicht else " (optioneel)")
        self.motivatie_box.layout.display = "flex"

    def _bevestig(self) -> None:
        if self.besloten or self.gekozen is None:
            return
        self._rond_af(self.motivatie.value)

    def _rond_af(self, motivatie: str) -> None:
        try:
            besluit = Besluit(voorstel_id=self.v.id, keuze=self.gekozen, motivatie=motivatie,
                              review_oordeel=self.v.review.oordeel)
        except ValueError as e:
            fouten = "; ".join(err["msg"].removeprefix("Value error, ") for err in e.errors())
            self.status.value = f'<span style="color:#B42318">{html.escape(fouten)}</span>'
            return
        self.besloten = True
        for knop in [*self.knoppen.values(), self.bevestig]:
            knop.disabled = True
        self.motivatie.disabled = True
        extra = f" · {html.escape(motivatie)}" if motivatie else ""
        self.status.value = f"<b>Besluit PO: {besluit.keuze}</b>{extra}"
        self.klaar(besluit)


class _Vraag:
    """Open vraag aan de PO over business of data."""

    def __init__(self, vraag: str, context: str, klaar: Callable[[str], None]):
        self.klaar = klaar
        self.verstuurd = False
        kop = w.HTML(CSS + f"""<div class="dsh-kaart" style="--dsh-k:#23395B">
          <div class="dsh-meta">Vraag aan de PO</div><h3>{html.escape(vraag)}</h3>
          {'<div>' + html.escape(context) + '</div>' if context else ''}</div>""")
        self.antwoord = w.Textarea(placeholder="Antwoord (of: weet ik niet)", layout=w.Layout(width="560px", height="70px"))
        self.knop = w.Button(description="Versturen", button_style="primary")
        self.status = w.HTML()
        self.knop.on_click(lambda _b: self._verstuur())
        self.widget = w.VBox([kop, self.antwoord, self.knop, self.status], layout=w.Layout(margin="0 0 16px"))

    def _verstuur(self) -> None:
        if self.verstuurd:
            return
        tekst = self.antwoord.value.strip()
        if not tekst:
            self.status.value = '<span style="color:#B42318">Geef een antwoord, of schrijf "weet ik niet".</span>'
            return
        self.verstuurd = True
        self.antwoord.disabled = self.knop.disabled = True
        self.status.value = "<b>Verstuurd.</b>"
        self.klaar(tekst)


class Cockpit:
    def __init__(self, log: Beslislog | None = None):
        self.log = log or Beslislog()
        self.feed = w.VBox()

    def _ipython_display_(self) -> None:
        from IPython.display import display
        display(self.feed)

    def _voeg_toe(self, widget: w.Widget) -> None:
        self.feed.children = (*self.feed.children, widget)

    def melding(self, tekst: str) -> None:
        self._voeg_toe(w.HTML(f'<div class="dsh-meta" style="margin:0 0 8px">{html.escape(tekst)}</div>'))

    # ---------- async ----------
    async def vraag_akkoord(self, v: Voorstel, figuur: Any = None) -> Besluit:
        fut = asyncio.get_running_loop().create_future()
        self._voeg_toe(_Kaart(v, lambda b: self._los_op(fut, v, b), figuur).widget)
        return await fut

    async def vraag_po(self, vraag: str, context: str = "") -> str:
        fut = asyncio.get_running_loop().create_future()
        self._voeg_toe(_Vraag(vraag, context, lambda t: fut.done() or fut.set_result(t)).widget)
        return await fut

    def _los_op(self, fut: asyncio.Future, v: Voorstel, b: Besluit) -> None:
        self.log.schrijf(v, b)
        if not fut.done():
            fut.set_result(b)

    def start(self, coro: Coroutine) -> asyncio.Task:
        """Start een workflow als achtergrondtaak; fouten verschijnen in de feed."""
        taak = asyncio.ensure_future(coro)

        def _klaar(t: asyncio.Task) -> None:
            if t.cancelled():
                self.melding("Workflow gestopt.")
            elif t.exception() is not None:
                tb = "".join(traceback.format_exception(t.exception()))
                self._voeg_toe(w.HTML(f'<pre style="color:#B42318">{html.escape(tb)}</pre>'))

        taak.add_done_callback(_klaar)
        return taak
