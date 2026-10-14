"""Generate the RF300 form and metrics from the same manifest served by the API."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.mushroom_model import PortableMushroom

model = PortableMushroom()
m = model.metadata
colors = dict(n='Bruin', b='Beige', g='Grijs', r='Groen', p='Roze', u='Paars', e='Rood',
              w='Wit', y='Geel', l='Blauw', o='Oranje', k='Zwart', f='Geen')
surfaces = dict(i='Vezelig', g='Gegroefd', y='Schubbig', s='Glad', h='Glanzend', l='Leerachtig',
               k='Zijdeachtig', t='Kleverig', w='Gerimpeld', e='Vlezig', d='Droog', f='Geen')
labels = {'cap_diameter': ('Hoeddiameter', 'cm'), 'stem_height': ('Steelhoogte', 'cm'),
          'stem_width': ('Steeldikte', 'mm'), 'cap_shape': ('Hoedvorm', ''),
          'gill_color': ('Lamelkleur', ''), 'season': ('Seizoen', ''),
          'stem_surface': ('Steeloppervlak', ''), 'ring_type': ('Ringtype', ''),
          'spore_print_color': ('Kleur sporenafdruk', ''), 'habitat': ('Leefomgeving', ''),
          'jumbled_noise_1': ('Extra datasetkenmerk (ruis)', '')}
maps = {'cap_shape': dict(b='Klokvormig', c='Kegelvormig', x='Bol', f='Vlak', s='Ingedrukt', p='Bolvormig', o='Anders'),
        'gill_color': colors, 'spore_print_color': colors, 'stem_surface': surfaces,
        'ring_type': dict(c='Spinnenwebachtig', e='Vergankelijk', r='Uitstaand', g='Gegroefd', l='Groot',
                          p='Hangend', s='Schedevormig', z='Zone', y='Schubbig', m='Beweegbaar', f='Geen'),
        'habitat': dict(g='Gras', l='Bladeren', m='Weide', p='Paden', h='Heide', u='Stedelijk', w='Afvalterrein', d='Bos'),
        'season': dict(s='Lente', u='Zomer', a='Herfst', w='Winter'), 'jumbled_noise_1': {}}
categories = {spec['name']: spec['categories'] for spec in m['categorical']}
order = ['cap_diameter', 'cap_shape', 'gill_color', 'stem_height', 'stem_width', 'season']
order += [name for name in m['features'] if name not in order]
fields = []
for name in order:
    label, unit = labels[name]
    field = {'name': name, 'label': label, 'unit': unit}
    if name in categories:
        field['options'] = [[code, maps[name].get(code, f'Datasetcode {code}')] for code in categories[name]]
    fields.append(field)
data = {key: m[key] for key in ['model_id', 'model_name', 'example', 'metrics',
        'dataset_rows', 'train_rows', 'test_rows', 'validation_note', 'source_pipeline_sha256']}
data['fields'] = fields
Path(__file__).with_name('metadata.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('Exported RF300 fields, example, source hash and matching validation metrics.')
