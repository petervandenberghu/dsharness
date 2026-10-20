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
.dsh-tabel{border-collapse:collapse;font-size:13px;width:100%}
.dsh-tabel th,.dsh-tabel td{text-align:left;vertical-align:top;padding:4px 8px;border-bottom:1px solid var(--jp-border-color2,#E3E7EC)}
.dsh-tabel th{color:var(--jp-ui-font-color2,#5B6577);font-weight:600}
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


class _Vragen:
    """Eén of meer vragen aan de PO, elk met een eigen antwoordveld en samen één knop."""

    def __init__(self, vragen: list[tuple[str, str]], klaar: Callable[[list[str]], None]):
        self.klaar = klaar
        self.verstuurd = False
        e = html.escape
        meer = len(vragen) > 1
        delen: list[w.Widget] = [w.HTML(CSS + f"""<div class="dsh-kaart" style="--dsh-k:#23395B">
          <div class="dsh-meta">{'Vragen' if meer else 'Vraag'} aan de PO{f' · {len(vragen)} deelvragen, elk een eigen antwoord' if meer else ''}</div>
          {'' if meer else f'<h3>{e(vragen[0][0])}</h3>' + (f'<div>{e(vragen[0][1])}</div>' if vragen[0][1] else '')}</div>""")]
        self.antwoorden = []
        for i, (vraag, context) in enumerate(vragen, 1):
            if meer:
                delen.append(w.HTML(f"<div style='margin:8px 0 2px'><b>{i}. {e(vraag)}</b>"
                                    + (f"<div class='dsh-meta'>{e(context)}</div>" if context else "") + "</div>"))
            veld = w.Textarea(placeholder="Antwoord (of: weet ik niet)", layout=w.Layout(width="560px", height="60px"))
            self.antwoorden.append(veld)
            delen.append(veld)
        self.knop = w.Button(description="Versturen", button_style="primary")
        self.status = w.HTML()
        self.knop.on_click(lambda _b: self._verstuur())
        self.widget = w.VBox([*delen, self.knop, self.status], layout=w.Layout(margin="0 0 16px"))

    def _verstuur(self) -> None:
        if self.verstuurd:
            return
        teksten = [a.value.strip() for a in self.antwoorden]
        if leeg := [str(i) for i, t in enumerate(teksten, 1) if not t]:
            self.status.value = (f'<span style="color:#B42318">Beantwoord ook vraag {", ".join(leeg)}, of schrijf "weet ik niet".</span>'
                                 if len(teksten) > 1 else '<span style="color:#B42318">Geef een antwoord, of schrijf "weet ik niet".</span>')
            return
        self.verstuurd = True
        for a in self.antwoorden:
            a.disabled = True
        self.knop.disabled = True
        self.status.value = "<b>Verstuurd.</b>"
        self.klaar(teksten)


def foutkaart(e: BaseException) -> str:
    """Korte uitleg bovenaan, de volledige traceback inklapbaar eronder."""
    from pydantic_ai.exceptions import ModelAPIError

    from .modellen import korte_fout

    modelfout = isinstance(e, ModelAPIError) or (isinstance(e, BaseExceptionGroup) and all(
        isinstance(x, ModelAPIError) for x in e.exceptions))
    if modelfout:
        kop = "De modellen zijn op dit moment niet beschikbaar"
        uitleg = ("Wat je kunt doen: wacht tot het quotum vernieuwd is (gratis Gemini: elke dag), of zet in project.yaml "
                  "een ander model voor deze rol. Voer daarna de startcel opnieuw uit: afgeronde fases worden overgeslagen.")
    else:
        kop, uitleg = "De workflow is gestopt door een fout", "De technische details staan hieronder."
    tb = "".join(traceback.format_exception(e))
    return CSS + f"""<div class="dsh-kaart" style="--dsh-k:#B42318">
      <div class="dsh-meta">Fout</div><h3>{html.escape(kop)}</h3>
      <p style="margin:0 0 6px">{html.escape(korte_fout(e))}</p><p class="dsh-meta" style="margin:0 0 6px">{html.escape(uitleg)}</p>
      <details><summary class="dsh-meta" style="cursor:pointer">Technische details</summary>
      <pre style="font-size:11px;white-space:pre-wrap;max-height:300px;overflow:auto">{html.escape(tb)}</pre></details></div>"""


class _PlanSamenvatting:
    """Kleine wrapper zodat het beslislog het F4-plan als rapport kan opslaan."""

    def __init__(self, **velden):
        self.velden = velden

    def model_dump(self, mode: str = "json") -> dict:
        return self.velden


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

    def _toon_profielen(self, r) -> None:
        e = html.escape
        kleur = {"geen_na": "", "MCAR": "#2C7A57", "MAR": "#A86300", "MNAR": "#B42318", "onbekend": "#56627A"}
        rijen = "".join(
            f"<tr><td><b>{e(p.kolom)}</b><div class='dsh-meta'>{p.type}</div></td>"
            f"<td>{p.na_aantal or ''}</td>"
            f"<td style='color:{kleur[p.na_mechanisme]};font-weight:600'>{'' if p.na_mechanisme == 'geen_na' else p.na_mechanisme}"
            f"<div class='dsh-meta' style='font-weight:400'>{e(p.na_onderbouwing)}</div></td>"
            f"<td>{e(p.verdeling)}</td><td>{e(p.uitschieters)}</td><td>{e(p.relatie_target)}</td></tr>"
            for p in r.profielen)
        punten = "".join(f"<li>{e(x)}</li>" for x in r.voor_f4)
        self._voeg_toe(w.HTML(CSS + f"""<div class="dsh-kaart" style="--dsh-k:#56627A;max-width:1100px">
          <div class="dsh-meta">F3 · Kolomprofielen (trainset)</div><h3>{e(r.samenvatting)}</h3>
          <p style="margin:0 0 8px"><b>Target:</b> {e(r.target)}</p>
          <div style="overflow-x:auto"><table class="dsh-tabel"><thead><tr><th>kolom</th><th>NA</th><th>mechanisme</th>
          <th>verdeling</th><th>uitschieters</th><th>relatie met target</th></tr></thead><tbody>{rijen}</tbody></table></div>
          <p style="margin:10px 0 4px"><b>Voor F4</b></p><ul style="margin:0;padding-left:18px">{punten}</ul></div>"""))

    def toon_plan(self, plan, proef: dict) -> None:
        from .acties import omschrijf
        e = html.escape
        self.log.schrijf_rapport("F4", _PlanSamenvatting(stappen=[omschrijf(s) for s in plan.stappen], **proef))
        stappen = "".join(f"<li>{e(omschrijf(s))}</li>" for s in plan.stappen)
        vangnet = (f"<p class='dsh-meta' style='margin:8px 0 0'>Vangnet (mediaan, alleen voor onverwachte NA in nieuwe data): "
                   f"{e(', '.join(proef['vangnet']))}</p>") if proef["vangnet"] else ""
        self._voeg_toe(w.HTML(CSS + f"""<div class="dsh-kaart" style="--dsh-k:#2C7A57">
          <div class="dsh-meta">F4 · Preparatie-pipeline (gefit op train)</div>
          <h3>Model-klaar: {proef['rijen']} rijen × {len(proef['kolommen'])} features</h3>
          <ol style="margin:0;padding-left:20px">{stappen}</ol>
          <p class="dsh-meta" style="margin:8px 0 0">Features: {e(', '.join(proef['kolommen']))}</p>{vangnet}</div>"""))

    def toon_resultaten(self, lab) -> None:
        e = html.escape
        self.log.schrijf_rapport("F5", _PlanSamenvatting(cv=lab.cv.model_dump(), gekozen=lab.gekozen,
                                                          resultaten=[r.model_dump() for r in lab.resultaten.values()]))
        beste = lab.beste()
        rijen = "".join(
            f"<tr style='{'font-weight:700' if r.naam == lab.gekozen else ''}'><td>{e(r.naam)}{' ✓' if r.naam == lab.gekozen else ''}</td>"
            f"<td>{e(r.omschrijving.split(': ', 1)[-1])}</td><td>{r.gemiddelde:.3f} ± {r.sd:.3f}</td><td>{r.train:.3f}</td>"
            f"<td>{'beste' if beste and r.naam == beste.naam else ''}"
            f"{'<span style=color:#A86300>= baseline: leert niets</span>' if lab.gelijk_aan_baseline(r) else ''}</td></tr>"
            for r in sorted(lab.resultaten.values(), key=lambda r: r.gemiddelde, reverse=lab.beter(1, 0)))
        kop = f"Gekozen voor F6: {lab.gekozen}" if lab.gekozen else f"{len(lab.resultaten)} experiment(en)"
        self._voeg_toe(w.HTML(CSS + f"""<div class="dsh-kaart" style="--dsh-k:#2F6FAF;max-width:1000px">
          <div class="dsh-meta">F5 · Cross-validatie op train · {e(lab.cv.methode)}, {lab.cv.folds} folds</div>
          <h3>{e(kop)}</h3><div style="overflow-x:auto"><table class="dsh-tabel"><thead><tr><th>experiment</th><th>model</th>
          <th>{e(lab.cv.metriek)} (validatie)</th><th>train</th><th></th></tr></thead><tbody>{rijen}</tbody></table></div>
          <p class="dsh-meta" style="margin:8px 0 0">Groot verschil tussen train en validatie wijst op overfitting.
          Een verschil kleiner dan de spreiding (±) is vooral ruis.</p></div>"""))

    def toon_eindtest(self, lab) -> None:
        e = html.escape
        r = lab.eind
        self.log.schrijf_rapport("F6", r)
        kleur = "#2C7A57" if r.binnen_verwachting else "#A86300"
        oordeel = ("binnen de verwachting van de cross-validatie" if r.binnen_verwachting else
                   "buiten de verwachting van de cross-validatie (bij weinig testrijen kan dat toeval zijn)")
        extra = ", ".join(f"{k} {v:.3f}" for k, v in r.extra.items())
        base = f"<p style='margin:4px 0'>Baseline op test: {r.test_baseline:.3f}</p>" if r.test_baseline is not None else ""
        self._voeg_toe(w.HTML(CSS + f"""<div class="dsh-kaart" style="--dsh-k:{kleur}">
          <div class="dsh-meta">F6 · Eindtest · {r.n_test} testrijen, eenmalig · de testset is nu verbrand</div>
          <h3>'{e(r.experiment)}': {e(r.metriek)} = {r.test:.3f} op test</h3>
          <p style="margin:4px 0">Cross-validatie gaf {r.cv_gemiddelde:.3f} ± {r.cv_sd:.3f}: {oordeel}.</p>{base}
          <p class="dsh-meta" style="margin:6px 0 0">Overige maten op test: {e(extra)}</p></div>"""))

    # ---------- async ----------
    async def vraag_akkoord(self, v: Voorstel, figuur: Any = None) -> Besluit:
        fut = asyncio.get_running_loop().create_future()
        self._voeg_toe(_Kaart(v, lambda b: self._los_op(fut, v, b), figuur).widget)
        return await fut

    async def vraag_po(self, vraag: str, context: str = "") -> str:
        return (await self.vraag_po_meerdere([(vraag, context)]))[0]

    async def vraag_po_meerdere(self, vragen: list[tuple[str, str]]) -> list[str]:
        fut = asyncio.get_running_loop().create_future()

        def klaar(antwoorden: list[str]) -> None:
            for (vraag, _), antwoord in zip(vragen, antwoorden):
                self.log.schrijf_vraag(vraag, antwoord)
            if not fut.done():
                fut.set_result(antwoorden)

        self._voeg_toe(_Vragen(vragen, klaar).widget)
        return await fut

    def toon_rapport(self, rapport) -> None:
        if hasattr(rapport, "profielen"):
            self.log.schrijf_rapport("F3", rapport)
            self._toon_profielen(rapport)
            return
        self.log.schrijf_rapport("F1", rapport)
        e = html.escape
        items = "".join(
            f"<li><b>{e(b.kolom or 'algemeen')}</b> ({e(b.soort)}): {e(b.beschrijving)}"
            f"<div class='dsh-meta'>{e(b.bewijs)}</div></li>" for b in rapport.bevindingen)
        self._voeg_toe(w.HTML(CSS + f"""<div class="dsh-kaart" style="--dsh-k:#56627A">
          <div class="dsh-meta">Verkenningsrapport · {len(rapport.voorstellen)} voorstel(len) volgen</div>
          <h3>{e(rapport.samenvatting)}</h3><ul style="margin:0;padding-left:18px">{items}</ul></div>"""))

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
                self._voeg_toe(w.HTML(foutkaart(t.exception())))

        taak.add_done_callback(_klaar)
        return taak
