const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.join(__dirname, '.generated', 'preview_frontend');
const policy = require(path.join(root, 'mushroom.js'));
const config = require(path.join(root, 'config.js'));
const metadata = JSON.parse(fs.readFileSync(path.join(root, 'metadata.json'), 'utf8'));

test('preview uses its own local origin, preserving the original API URL logic', () => {
  assert.equal(config.resolveApiUrl({hostname:'127.0.0.1',port:'8011',protocol:'http:',origin:'http://127.0.0.1:8011'}),'http://127.0.0.1:8011');
});

test('frontend accepts the proposed threshold and never confirms edibility', () => {
  const threshold = metadata.decision_threshold;
  for (const score of [0,threshold-.0001,threshold,threshold+.0001,.4999,1]) {
    const view = policy.present({label:score>=.5?'p':'e',probability_poisonous:score,
      decision_threshold:threshold,decision_policy_id:metadata.decision_policy_id,
      decision:score>=threshold?'suspected_poisonous':'uncertain'});
    assert.equal(view.title,score>=threshold?'Model voorspelt: giftig':'Onzeker / mogelijk giftig');
    assert.notEqual(view.title,'eetbaar');
  }
});

test('missing, malformed and contradictory thresholds or policies are rejected', () => {
  for (const decision_threshold of [undefined,NaN,0,-.1,1,'0.45']) {
    assert.throws(()=>policy.present({label:'e',probability_poisonous:.4,
      decision_threshold,decision:'uncertain',decision_policy_id:policy.POLICY_ID}));
  }
  assert.throws(()=>policy.present({label:'e',probability_poisonous:.49,
    decision_threshold:metadata.decision_threshold,decision:'uncertain',decision_policy_id:policy.POLICY_ID}));
  assert.throws(()=>policy.present({label:'e',probability_poisonous:.4,
    decision_threshold:metadata.decision_threshold,decision:'uncertain',decision_policy_id:'old'}));
});

test('the same eleven fields and accurately scoped metrics are delivered', () => {
  assert.equal(metadata.fields.length,11);
  assert.deepEqual(new Set(metadata.fields.map(x=>x.name)),new Set(Object.keys(metadata.example)));
  const html = fs.readFileSync(path.join(root,'index.html'),'utf8');
  assert.ok(html.includes('CONCEPT MILAN'));
  assert.ok(html.includes('Geen nieuwe onafhankelijke eindtest'));
  assert.ok(!html.includes('Resultaten op 900 testrecords'));
  assert.ok(!html.includes('oorspronkelijke modelclassificatie met de 50%-grens'));
});
