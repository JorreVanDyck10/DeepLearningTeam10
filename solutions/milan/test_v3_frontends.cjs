const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const folders = ['preview_frontend'];
for (const [receipt, folder] of [['candidate_package.json','preview_frontend_v3'], ['voting_package.json','preview_frontend_voting_v3']]) {
  if (JSON.parse(fs.readFileSync(path.join(__dirname,'results_v3',receipt),'utf8')).packaged) folders.push(folder);
}
for (const folder of folders) {
  const root = path.join(__dirname,'.generated',folder);
  const policy = require(path.join(root,'mushroom.js'));
  const metadata = JSON.parse(fs.readFileSync(path.join(root,'metadata.json'),'utf8'));
  const threshold = metadata.decision_threshold;
  test(`${folder}: decision boundary and raw labels remain separate`, () => {
    assert.equal(policy.POLICY_ID,metadata.decision_policy_id);
    for (const p of [0,threshold-.0001,threshold,threshold+.0001,.5,1]) {
      const result={label:p>.5?'p':'e',probability_poisonous:p,decision_threshold:threshold,
        decision_policy_id:policy.POLICY_ID,decision:p>=threshold?'suspected_poisonous':'uncertain'};
      const presented=policy.present(result);
      assert.equal(presented.title,p>=threshold?'Model voorspelt: giftig':'Onzeker / mogelijk giftig');
      assert.notEqual(presented.title,'eetbaar');
      assert.throws(()=>policy.present({...result,decision:result.decision==='uncertain'?'suspected_poisonous':'uncertain'}));
    }
  });
  test(`${folder}: bad threshold and stale policy are rejected`, () => {
    const result={label:'e',probability_poisonous:0,decision_threshold:threshold,
      decision_policy_id:policy.POLICY_ID,decision:'uncertain'};
    for (const value of [undefined,NaN,0,-1,1,'0.45']) assert.throws(()=>policy.present({...result,decision_threshold:value}));
    assert.throws(()=>policy.present({...result,decision_policy_id:'old'}));
  });
  test(`${folder}: the original eleven-field form and honest CV text`, () => {
    assert.equal(metadata.fields.length,11);
    assert.deepEqual(new Set(metadata.fields.map(f=>f.name)),new Set(Object.keys(metadata.example)));
    const html=fs.readFileSync(path.join(root,'index.html'),'utf8');
    assert.ok(html.includes('CONCEPT MILAN'));
    assert.ok(html.includes('Geen nieuwe onafhankelijke eindtest'));
    assert.ok(!html.includes('Resultaten op 900 testrecords'));
  });
}
