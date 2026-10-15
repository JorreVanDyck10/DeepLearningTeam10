"""Local Mushroom preview: reuse the current input schema and model reader.

Only solutions/milan/.generated/preview_frontend is written. The proposal is
explicitly applied here; neither the current backend nor the website is edited.
Install requirements-preview.txt and run python solutions/milan/preview.py.
"""
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGES = HERE / '.generated/preview_packages'
if PACKAGES.exists():
    sys.path.insert(0, str(PACKAGES))
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from backend.main import MushroomInput, MushroomPrediction, predict_mushroom
from backend.mushroom_model import PortableMushroom, MODEL_ID

POLICY_ID = 'milan_macro_f1_threshold_preview_v2'


def prepare_frontend(metadata, *, target=None, policy_id=POLICY_ID):
    target = Path(target) if target is not None else HERE / '.generated/preview_frontend'
    assert target.resolve().is_relative_to(HERE)
    shutil.copytree(ROOT / 'frontend', target, dirs_exist_ok=True)
    original = json.loads((target / 'metadata.json').read_text(encoding='utf-8'))
    for key in ['model_id', 'model_name', 'source_pipeline_sha256', 'example',
                'train_rows', 'test_rows', 'dataset_rows', 'validation_note']:
        original[key] = metadata[key]
    policy_metrics = metadata['proposed_policy_metrics']
    original['metrics'] = {'accuracy': policy_metrics['accuracy_mean'],
        'recall_poisonous': policy_metrics['recall_p_mean'],
        'poisonous_predicted_edible': int(policy_metrics['fn_total'])}
    original['decision_threshold'] = metadata['proposed_decision_threshold']
    original['decision_policy_id'] = policy_id
    # Original user-facing categories remain. The internal missing token is not
    # offered as if it were a real mushroom measurement.
    learned = {c['name']: set(c['categories']) for c in metadata['categorical']}
    for field in original['fields']:
        if 'options' in field:
            assert {v[0] for v in field['options']} <= learned[field['name']]
    (target / 'metadata.json').write_text(json.dumps(original, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    html_path = target / 'index.html'
    html = html_path.read_text(encoding='utf-8')
    old_description = 'De geselecteerde Random Forest heeft 300 bomen, maximale diepte 20 en gebalanceerde klassegewichten. Het formulier gebruikt de elf kenmerken van de gekozen pipeline. Het eerste ruisveld is verwijderd; het tweede is optioneel.'
    if metadata.get('serving_format') == 'milan_native_voting_v1':
        names = {'rf_selection': 'Random Forest', 'lightgbm': 'LightGBM', 'catboost': 'CatBoost'}
        active = [names[c['family']] for c in metadata['components']]
        description = 'Dit concept combineert ' + ', '.join(active) + ' via gewogen voting. Het formulier behoudt dezelfde elf invoervelden.'
    elif metadata.get('ignored_features'):
        names = {'spore_print_color': 'sporenkleur', 'jumbled_noise_1': 'extra ruisveld'}
        unused = ', '.join(names[f] for f in metadata['ignored_features'])
        description = 'Dit concept gebruikt een Random Forest met interne featureselectie. Het formulier behoudt dezelfde elf invoervelden. Niet gebruikt door dit model: ' + unused + '.'
    else:
        description = old_description
    assert old_description in html, 'Original model description changed: review before copying.'
    html = html.replace(old_description, description)
    default_policy = metadata['proposed_policy'] == 'default'
    threshold_note = ('De beslisgrens is de vooraf vaste standaardgrens 0,5.' if default_policy else
        'De beslisdrempel werd uitsluitend binnen training gekozen.')
    old = 'Resultaten op 900 testrecords uit het cursusbestand van 5.000 records; 4.100 records zijn gebruikt voor training. Deze testset is tijdens het onderzoek al bekeken. Het model herkent nog niet alle giftige paddenstoelen correct.'
    assert old in html, 'Original frontend changed: review the preview text before copying.'
    html = html.replace(old,
        'CONCEPT MILAN: resultaten van vijf buitenfolds op 4.100 ontwikkelingsrecords. ' + threshold_note +
        ' Geen nieuwe onafhankelijke eindtest; geen bevestiging van eetbaarheid. Citi Bike is in deze lokale preview niet aangesloten.')
    html = html.replace('Deze metrics beschrijven de oorspronkelijke modelclassificatie met de 50%-grens. Het formulier gebruikt een voorzichtiger beslisregel en geeft geen bevestiging van eetbaarheid.',
        ('Deze metrics beschrijven de standaardbeslissing bij 0,5.' if default_policy else
         'Deze metrics beschrijven de binnen training afgestemde beslispolicy.') +
        ' De ruwe classifierlabels blijven afzonderlijk beschikbaar. Het formulier geeft geen bevestiging van eetbaarheid.')
    (target / 'index.html').write_text(html, encoding='utf-8')
    js_path = target / 'mushroom.js'
    js = js_path.read_text(encoding='utf-8')
    js = js.replace('mushroom_no_edibility_confirmation_v1', policy_id)
    js = js.replace('const expected = score >= 0.5 ?',
        'const threshold = result.decision_threshold;\n    if (!Number.isFinite(threshold) || threshold <= 0 || threshold >= 1) throw new Error("Ongeldige conceptdrempel.");\n    const expected = score >= threshold ?')
    (target / 'mushroom.js').write_text(js, encoding='utf-8')
    return target


def create_app(manifest=None, *, policy_id=POLICY_ID, preview_folder=None, model_reader=PortableMushroom):
    manifest = Path(manifest) if manifest is not None else HERE / 'artifacts/mushroom_rf300.json'
    model = model_reader(manifest)
    metadata = model.metadata
    folder = prepare_frontend(metadata, target=preview_folder, policy_id=policy_id)
    app = FastAPI(title='Milan — lokale Mushroom-conceptcontrole')
    app.state.mushroom_model = model
    threshold = metadata['proposed_decision_threshold']

    class PreviewPrediction(MushroomPrediction):
        decision_threshold: float

    @app.post('/predict/mushroom', response_model=PreviewPrediction)
    def predict(payload: MushroomInput, request: Request):
        # Call the real controller (and its normal request/error handling), then
        # apply the proposed decision policy explicitly in this preview only.
        result = predict_mushroom(payload, request)
        score = result['probability_poisonous']
        result.update(decision='suspected_poisonous' if score >= threshold else 'uncertain',
            prediction='Model voorspelt: giftig' if score >= threshold else 'Onzeker / mogelijk giftig',
            decision_policy_id=policy_id, decision_threshold=threshold)
        # Keep the original raw label from .5; the policy decision is separate.
        return result

    @app.get('/health')
    def health():
        return {'status': 'preview_only', 'model_id': MODEL_ID,
            'mushroom_model_loaded': True, 'model_revision': metadata['source_pipeline_sha256'],
            'mushroom_model_id': MODEL_ID, 'mushroom_model_revision': metadata['source_pipeline_sha256'],
            'citibike_model_loaded': False}

    app.mount('/', StaticFiles(directory=folder, html=True))
    return app


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(create_app(), host='127.0.0.1', port=8011)
