# dsharness

Multi-agent data science harness met een cockpit-notebook. De PO (gebruiker) keurt elk voorstel goed, past het aan, wijst het af of overrulet het. Alle besluiten en vragen komen in een beslislog.

**Status:** F1 (eerste blik) → F2 (vaste regels) → split werkt, met de Data Scout en de Reviewer.

## Starten
```
uv sync
uv run jupyter lab 00_cockpit.ipynb
```
Instellingen staan in `project.yaml`: doel, target, data en het model per rol. In VS Code kies je `.venv` als kernel.

Modellen geef je op als `provider:model`, bijvoorbeeld `google:gemini-flash-latest` of `ollama:gemma4:12b`.
Met `data_policy: local_only` weigert de harness cloudmodellen.

## Opbouw
| Bestand | Rol |
|---|---|
| `data.py` | DataLaag: de enige toegang tot de data. Na de split zit de testset in de kluis |
| `verkennen.py` | Verkenningsfuncties: compacte samenvattingen, geen ruwe rijen |
| `scout.py` | Data Scout (Pydantic AI): verkent, vraagt de PO, stelt vaste regels en een split voor |
| `reviewer.py` | Reviewer: vaste regels in code plus optioneel een LLM, die alleen strenger mag zijn |
| `workflow.py` | Volgorde en gates in gewone Python: review → herzien → PO → uitvoeren |
| `cockpit.py` | Kaarten in het notebook (ipywidgets) |
| `datasets.py` | Oefendatasets met bekende valkuilen: voor tests, evals en onderwijs |
| `modellen.py` | Modellen uit een naam, met wachten bij rate limits (elke provider) |

## Tests en evals
```
uv run pytest                                   # zonder netwerk
uv run python scripts/eval_scout.py <scout-model> <reviewer-model|->
```
De eval laat de Scout los op de vier oefendatasets, met een gesimuleerde PO. Hij meet welke valkuilen gevonden worden en of de split klopt.

## Ontwerpkeuze: wachten op de PO
Een workflow draait als achtergrondtaak (`cockpit.start(...)`) en wacht met `await`. `scripts/check_async.py` laat zien waarom:

| | ipykernel 6 | ipykernel 7 |
|---|---|---|
| achtergrondtaak | ✓ | ✓ |
| `await` direct in een cel | hangt | ✓ |
| jupyter-ui-poll (blokkerend) | ✓ | ✗ |
