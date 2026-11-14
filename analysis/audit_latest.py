import sys, json, hashlib, collections
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fcct.report import binary_metrics, pope_answer

base = ROOT / 'analysis/latest_archive/outputs/fcct'
load = lambda p: json.loads(p.read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rows = [json.loads(s) for s in (base/'pope.jsonl').open()]
traces = [json.loads(s) for s in (base/'trace.jsonl').open()]
profile = load(base/'profile.json')
comparison = load(base/'report/comparison.json')
tm, pm = load(base/'trace.meta.json'), load(base/'pope.meta.json')
rng = np.random.default_rng(17)
out = {'archive_sha256': digest(ROOT/'outputs.zip'), 'questions': len(rows),
       'unique_sample_ids': len({r['sample_id'] for r in rows}),
       'evaluation_images': len({r['image_id'] for r in rows}),
       'trace_rows':len(traces), 'trace_unique_ids':len({r['sample_id'] for r in traces}),
       'trace_states':dict(collections.Counter(r['status'] for r in traces)),
       'image_overlap':len({r['image_id'] for r in traces}&{r['image_id'] for r in rows}),
       'integrity':{'predictions_hash':digest(base/'pope.jsonl') == comparison['hashes']['predictions'],
                    'profile_hash':digest(base/'profile.json') == comparison['hashes']['profile'],
                    'trace_hash':digest(base/'trace.jsonl') == profile['source_trace_sha256'],
                    'trace_meta_hash':digest(base/'trace.meta.json') == profile['source_meta_sha256'],
                    'evaluation_profile_hash':digest(base/'profile.json') == pm['extra']['profile_sha256'],
                    'trace_eval_code_agree':tm['code_sha256'] == pm['code_sha256']},
       'splits':{}}
for split in ('random','popular','adversarial'):
    rs=[r for r in rows if r['split']==split]
    metrics={m:binary_metrics(rs,m) for m in ('regular','iri')}
    assert metrics == comparison['metrics'][split]
    switches=collections.Counter()
    by_image=collections.defaultdict(list)
    for r in rs:
        a=pope_answer(r['answers']['regular'])==r['label']
        b=pope_answer(r['answers']['iri'])==r['label']
        switches['corrected' if b and not a else 'harmed' if a and not b else 'both_correct' if a else 'both_wrong']+=1
        by_image[r['image_id']].append(int(b)-int(a))
    sums=np.array([sum(v) for v in by_image.values()]); ns=np.array([len(v) for v in by_image.values()])
    idx=rng.integers(len(sums),size=(10000,len(sums)))
    ci=np.quantile(100*sums[idx].sum(1)/ns[idx].sum(1),[.025,.975]).tolist()
    out['splits'][split]={'metrics':metrics,'switches':dict(switches),'accuracy_delta_cluster_bootstrap_95ci_pp':ci}
complete=[r for r in traces if r['status']=='complete']
gaps=np.array([r['probability_gap'] for r in complete])
out['gap_abs_quantiles']=dict(zip(['min','p1','p5','median','p95','max'],np.quantile(abs(gaps),[0,.01,.05,.5,.95,1]).tolist()))
out['gap_below_threshold']={str(x):int((abs(gaps)<x).sum()) for x in (1e-6,1e-5,1e-4,.001,.01)}
out['layer_analysis']={}
for component in ('attention','mlp'):
    image_values=collections.defaultdict(list)
    for r in complete:
        patches={p['layer']:p for p in r['patches'] if p['component']==component and p['group']=='last'}
        image_values[r['image_id']].append([patches[i]['rr'] for i in range(32)])
    mat=np.array([np.mean(v,axis=0) for v in image_values.values()])
    mean=mat.mean(0)
    assert np.allclose(mean,profile['rr'][component])
    rank=lambda v:np.argsort(-v,kind='stable')[:3].tolist()
    top=rank(mean)
    extreme=[]
    for layer in top:
        j=int(np.argmax(abs(mat[:,layer])))
        extreme.append({'layer':layer,'mean_rr':float(mean[layer]),'largest_abs_image_mean_rr':float(mat[j,layer]),
                        'that_image_id':list(image_values)[j], 'mean_without_that_image':float(np.delete(mat[:,layer],j).mean())})
    keep=np.argsort(abs(mat),axis=0)[0:int(.99*len(mat))]
    trimmed=np.take_along_axis(mat,keep,axis=0).mean(0)
    boot=mat[rng.integers(len(mat),size=(2000,len(mat)))].mean(1)
    selected=np.argsort(-boot,axis=1)[:,:3]
    out['layer_analysis'][component]={'raw_top3':top,'median_top3':rank(np.median(mat,axis=0)),
        'remove_top_1percent_abs_per_layer_top3':rank(trimmed), 'extreme_image_influence':extreme,
        'bootstrap_top3_inclusion':{str(i):float((selected==i).any(1).mean()) for i in top},
        'effective_source_to_target_edges':{str(s):[t for t in sorted(np.argsort(-mean)[:10].tolist()) if t>s] for s in top}}
(ROOT/'analysis/latest_audit_metrics.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
