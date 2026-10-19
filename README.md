# dsharness (prototype)

Multi-agent data science harness met een cockpit-notebook. De PO (gebruiker) keurt elk voorstel goed, past het aan, wijst het af of overrulet het. Alle besluiten komen in een beslislog.

Status: **prototype van de cockpit**. Er zijn nog geen LLM-agents.

## Starten
```
uv sync
uv run jupyter lab 00_cockpit.ipynb
```
In VS Code kies je `.venv` als kernel.

## Tests
```
uv run pytest
```
De kerneltests starten een echte kernel en bootsen klikken uit de frontend na.

## Ontwerpkeuze: wachten op de PO
Een workflow draait als achtergrondtaak (`cockpit.start(...)`) en wacht met `await cockpit.vraag_akkoord(...)`.
`scripts/check_async.py` laat zien waarom:

| | ipykernel 6 | ipykernel 7 |
|---|---|---|
| achtergrondtaak | ✓ | ✓ |
| `await` direct in een cel | hangt | ✓ |
| jupyter-ui-poll (blokkerend) | ✓ | ✗ |
