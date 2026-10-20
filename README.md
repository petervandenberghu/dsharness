# dsharness

Multi-agent data science harness met een cockpit-notebook. De PO (gebruiker) keurt elk voorstel goed, past het aan, wijst het af of overrulet het. Alle besluiten en vragen komen in een beslislog.

**Status:** F1 (eerste blik) → F2 (vaste regels) → split → F3 (verkennen op train) → F4 (preparatie-pipeline) → F5 (modelleren met cross-validatie) → F6 (eenmalige eindtest) → F7 (resultaat-notebook) werkt: de eerste versie is compleet.

## Starten
```
uv sync
uv run jupyter lab 00_cockpit.ipynb
```
Instellingen staan in `project.yaml`: doel, target, data en het model per rol. In VS Code kies je `.venv` als kernel.

Modellen geef je op als `provider:model`, bijvoorbeeld `google:gemini-3-flash-preview` of `ollama:gemma4-16k`, of als lijst: het volgende model neemt het over bij aanhoudende fouten.
Met `data_policy: local_only` weigert de harness cloudmodellen.

## Opbouw
| Bestand | Rol |
|---|---|
| `data.py` | DataLaag: de enige toegang tot de data. Na de split zit de testset in de kluis |
| `verkennen.py` | Verkenningsfuncties: compacte samenvattingen, geen ruwe rijen |
| `tools.py` | Gedeelde tools; F3-tools (NA-patroon, verdeling, relatie met target) werken alleen na de split |
| `scout.py` | Data Scout (F1): verkent, vraagt de PO, stelt vaste regels en een split voor |
| `prep.py` | Prep Engineer (F4): stelt preparatiestappen voor tot de pipeline model-klaar is |
| `prep_plan.py` | Goedgekeurde F4-stappen → één sklearn-Pipeline; volgorde per kolom ligt vast, fit alleen op train |
| `modeler.py` | Modeler (F5): CV-opzet, baseline, experimenten en modelkeuze, in rondes |
| `lab.py` | Voert F5 uit: cross-validatie op train met de F4-pipeline binnen elke fold |
| `verslag.py` | Alles van een project in één JSON; `reproduceer()` bouwt de keten opnieuw op en vergelijkt de getallen |
| `resultaat.py` | F7: notebook uit het verslag, uitgevoerd met papermill, plus HTML, model card en model.joblib |
| `contract.py` | Datacontract uit train: fouten (weigeren) en waarschuwingen (wel voorspellen) |
| `serve.py` / `cli.py` | FastAPI-service, rooktest en de opdracht `dsharness` |
| `tracking.py` | MLflow: experimenten, eindtest en model-registry, lokaal in de resultaatmap |
| `verkenner.py` | Verkenner (F3): kolomprofielen op train; NA-mechanisme pas na NA-patroon én een vraag aan de PO |
| `reviewer.py` | Reviewer: vaste regels in code plus optioneel een LLM, die alleen strenger mag zijn |
| `workflow.py` | Volgorde en gates in gewone Python: review → herzien → PO → uitvoeren |
| `cockpit.py` | Kaarten in het notebook (ipywidgets) |
| `datasets.py` | Oefendatasets met bekende valkuilen: voor tests, evals en onderwijs |
| `modellen.py` | Modellen uit een naam, met wachten bij rate limits (elke provider) |

## Resultaat (F7)
Na de eindtest maakt de harness in `resultaat/`: `07_resultaat.ipynb` (uitgevoerd) en `07_resultaat.html`, `model_card.md`,
`verslag.json` en `model.joblib`. Het notebook reproduceert de hele keten uit het verslag en controleert dat dezelfde
getallen terugkomen; klopt dat niet, dan faalt het. De tests controleren ook dat een vervalst verslag wordt betrapt.

## Deploy en MLflow (versie 2)
F7 schrijft naast het notebook ook `contract.json` (wat het model als invoer verwacht, uit train), `voorbeeld.json`
en een MLflow-database met alle experimenten, de eindtest en het model in de registry.
```
uv run dsharness serve resultaat            # API op http://127.0.0.1:8000/docs
uv run dsharness rooktest resultaat         # API geeft hetzelfde als het model, ongeldige invoer wordt geweigerd
uv run mlflow ui --backend-store-uri sqlite:///resultaat/mlflow.db
```
- `POST /voorspel` met `{"rijen": [...]}`. Een ontbrekende kolom of een verkeerd type geeft HTTP 422; een onbekende
  categorie, een waarde buiten het bereik van train of een ontbrekende waarde geeft een waarschuwing, wel een voorspelling.
- Het resultaat-notebook draait de rooktest mee.
- MLflow slaat het model op met cloudpickle, net als `model.joblib`: laad het alleen uit een bron die je vertrouwt.

## Tests en evals
```
uv run pytest                                   # zonder netwerk
uv run python scripts/eval_scout.py <scout-model> <reviewer-model|->
uv run python scripts/eval_verkenner.py <model> [datasets]
uv run python scripts/eval_prep.py <model> [datasets]
uv run python scripts/eval_modeler.py <model> [datasets]
```
De eval laat de Scout los op de vier oefendatasets, met een gesimuleerde PO. Hij meet welke valkuilen gevonden worden en of de split klopt.

## Ontwerpkeuze: wachten op de PO
Een workflow draait als achtergrondtaak (`cockpit.start(...)`) en wacht met `await`. `scripts/check_async.py` laat zien waarom:

| | ipykernel 6 | ipykernel 7 |
|---|---|---|
| achtergrondtaak | ✓ | ✓ |
| `await` direct in een cel | hangt | ✓ |
| jupyter-ui-poll (blokkerend) | ✓ | ✗ |
