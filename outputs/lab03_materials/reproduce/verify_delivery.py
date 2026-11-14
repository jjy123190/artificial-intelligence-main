"""Read-only verification of Lab 3 report data, graphs, trace replay and PDF/DOCX.
Run from the project root. Writes only its verification record under this delivery.
"""
from pathlib import Path
import sys, json, csv, hashlib, datetime, math, re, statistics, argparse
from xml.etree import ElementTree as ET
from delivery_files import CHECKSUM_FILE, delivery_files
from collections import defaultdict
BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parents[1];EV=BASE/'evidence'
sys.path.insert(0,str(ROOT))
from experiments.lab03.domain import Graph,Query,State,expert_legal,digest
from experiments.lab03.search import search
from experiments.lab03.evaluate import summarize,paired_effects
from docx import Document
from PIL import Image
import fitz
load=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def near(a,b,path=''):
 if isinstance(a,dict):
  assert set(a)==set(b),(path,'keys',set(a)^set(b))
  for k in a:near(a[k],b[k],path+'/'+str(k))
 elif isinstance(a,list):
  assert len(a)==len(b),(path,len(a),len(b))
  for i,(x,y) in enumerate(zip(a,b)):near(x,y,path+f'/{i}')
 elif isinstance(a,float):assert b is not None and abs(a-b)<1e-10,(path,a,b)
 else:assert a==b,(path,a,b)

def oracle_independent(gd,qd):
 # Uses plain dictionaries and loops, not domain.enumerate_paths or graph.goal.
 nodes={n['id']:n for n in gd['nodes']};out=defaultdict(list)
 for e in gd['edges']:out[e['source']].append(e)
 paths=[]
 def visit(ns,es):
  d=len(es)
  if d==len(qd['relations']):
   n=nodes[ns[-1]]
   if n['kind']=='Company' and n['sector']==qd['sector'] and n['listed']==qd['listed']:paths.append((tuple(ns),tuple(es)))
   return
  for e in out[ns[-1]]:
   if (e['relation']==qd['relations'][d] and e['target'] not in ns and e['verified'] and e['active'] and e['confidence']>=.55):
    visit(ns+[e['target']],es+[e['id']])
 visit([qd['start']],[]);return set(paths)

def read_rows():
 rs=list(csv.DictReader((EV/'raw/runs.csv').open(encoding='utf-8-sig',newline='')))
 ints=('seed','budget','hops_requested','expanded','total_work','successor_checks','heuristic_evaluations','heuristic_probe_states','proof_pruned','proof_checks','adjacency_edge_scans','heuristic_edge_checks','total_edge_checks','successor_cache_hits','successor_cache_entries','expert_rejections')
 for r in rs:
  for k in ints:r[k]=int(r[k])
  for k in ['valid','found','answerable']:assert r[k] in ['True','False'];r[k]=r[k]=='True'
  for k in ['redundancy_ratio','redundant_path_ratio','elapsed_ms']:r[k]=float(r[k])
  for k in ['hops','peak_frontier']:r[k]=int(r[k]) if r[k] else None
 return rs

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--review-updated-files',action='store_true',help='Explicitly review a rebuilt delivery without accepting an outdated seal. Re-sealing remains a separate step.')
 args=parser.parse_args()
 inventory=delivery_files()
 if not args.review_updated_files:
  assert CHECKSUM_FILE.is_file(),'Missing checksum manifest. Use --review-updated-files and explicit sealing only for reviewed rebuilds.'
  sealed=load(CHECKSUM_FILE)
  assert sealed['algorithm']=='sha256' and sealed['file_count']==len(sealed['files'])
  assert set(inventory)==set(sealed['files']),('delivery_inventory_changed',set(inventory)^set(sealed['files']))
  for rel,expected in sealed['files'].items():assert sha(inventory[rel])==expected,('attachment_modified',rel)
 src=load(EV/'source_manifest.json');source_count=0
 for s in src['sources']:
  path=ROOT/s['source'];assert path.is_file(),str(path);assert sha(path)==s['sha256'],str(path)
  if s.get('attachment'):assert sha(BASE/s['attachment'])==s['sha256'],s['attachment']
  source_count+=1
 protected=load(BASE/'reproduce/qa/protected_before.json')['hashes']
 for rel,expected in protected.items():assert (ROOT/rel).is_file() and sha(ROOT/rel)==expected,('protected_changed',rel)
 for rel,expected in src['core_hashes_at_report_time'].items():assert sha(ROOT/rel)==expected,rel
 rows=read_rows();assert len(rows)==3888
 summary=summarize(rows);near(summary,load(EV/'raw/summary.json'));near(paired_effects(rows),load(EV/'raw/paired_effects.json'))
 metrics=load(EV/'report_metrics.json');near(metrics['summary'],summary)
 index=load(EV/'case_index.json');cases=[]
 confirmation=load(EV/'raw/reliability/confirmation.json')
 confirmation_map={(x['query_id'],x['method'],x['budget']):x for x in confirmation['rows']+confirmation['negative_rows']}
 controlled=load(EV/'raw/hallucination_stress.json')['counterexamples']['budget_trap']
 for item in index['cases']:
  obj=load(BASE/item['file']);g=Graph.from_dict(obj['graph']);q=Query(**{**obj['query'],'relations':tuple(obj['query']['relations'])});params=obj['parameters']
  r=search(g,q,params['method'],params['budget'],params['weights'],params['model'],params['seed'],expert=params['expert'],llm_scores=params['llm_scores'],trace=True)
  expected=obj['result'];near({k:v for k,v in r.items() if k!='elapsed_ms'},{k:v for k,v in expected.items() if k!='elapsed_ms'},item['case_id'])
  saved_confirmation=confirmation_map.get((q.id,params['method'],params['budget']))
  if saved_confirmation:
   for key,value in saved_confirmation['after'].items():near(expected[key],value,item['case_id']+'/confirmation/'+key)
  if item['case_id'] in ['F10_controlled_neutral','F11_controlled_wrong']:
   frozen=controlled['neutral_scores' if item['case_id']=='F10_controlled_neutral' else 'wrong_scores']
   for key,value in frozen.items():near(expected[key],value,item['case_id']+'/controlled/'+key)
  independent=oracle_independent(obj['graph'],obj['query'])
  assert len(independent)==obj['independent_oracle']['valid_path_count']
  assert independent=={(tuple(x['nodes']),tuple(x['edge_ids'])) for x in obj['independent_oracle']['valid_paths']}
  if expected['found']:assert expected['valid']==((tuple(expected['path']),tuple(expected['edge_ids'])) in independent)
  assert digest(obj['graph'])==obj['graph_sha256'];assert digest(obj['query'])==obj['query_sha256']
  for e in expected['trace']:
   assert len(e['path'])==len(e['edge_ids'])+1
   for a,b,eid in zip(e['path'],e['path'][1:],e['edge_ids']):assert (g.edges[eid].source,g.edges[eid].target)==(a,b)
   assert e['depth']==len(e['edge_ids'])
  cases.append({'case_id':item['case_id'],'trace_events':len(expected['trace']),'independent_oracle_paths':len(independent),'result_match':True})
 # Figure spatial correctness and trace-state correspondence.
 figs=load(EV/'figure_index.json')
 for f in figs:
  png=BASE/'figures'/(f['name']+'.png');w,h=f['viewBox'];im=Image.open(png);assert im.size==(w*2,h*2)
  if f.get('case_id'):
   obj=load(EV/'cases'/(f['case_id']+'.json'));keys={tuple(e['edge_ids']) for e in obj['result']['trace']}
   assert keys=={tuple(n['edge_prefix']) for n in f['nodes']}
   for i,n in enumerate(f['nodes']):
    assert 0<=n['x'] and 0<=n['y'] and n['x']+n['w']<=w and n['y']+n['h']<=h
    for k in f['nodes'][i+1:]:assert not (n['x']<k['x']+k['w'] and n['x']+n['w']>k['x'] and n['y']<k['y']+k['h'] and n['y']+n['h']>k['y']),('overlap',f['name'])
 # DOCX text and all figures. Validate published primary numerical tables against raw JSON.
 doc=Document(BASE/'第31组_实验三_实验报告.docx');structure=load(EV/'report_structure.json')
 assert len(doc.tables)==25 and len(structure['tables'])==25 and len(doc.inline_shapes)==14
 # Member duties are a plan, with no numerical allocation or invented completion claims.
 team=doc.tables[-1]
 assert [c.text for c in team.rows[0].cells]==['成员','主要负责模块','具体职责与协作事项']
 assert [r.cells[0].text for r in team.rows[1:]]==['梁镇贤\n（组长）','陈宗波','温文辉','陈宇冲','刘亦达']
 md=(BASE/'第31组_实验三_实验报告.md').read_text(encoding='utf-8')
 team_text=md.split('## 附录 B 成员分工与协作安排',1)[1]
 assert '拟定成员职责' in team_text and 'B.2 协作与交叉复核' in team_text
 assert not any(word in team_text for word in ('%','％','工作量','占比','比例','已完成'))
 # The copied dataset statistics must agree with source graphs, not just stored tables.
 dataset=load(ROOT/'data/lab03/benchmark.json')
 records=dataset['records'] if isinstance(dataset,dict) else dataset
 for cells,split in zip([r.cells for r in doc.tables[1].rows[1:]],('train','validation','test','ood')):
  selected=[r for r in records if r['split']==split]
  hops=[sum(len(q['relations'])==hop for r in selected for q in r['queries']) for hop in (2,3,4)]
  expected=[split,str(len(selected)),str(sum(len(r['queries']) for r in selected)),str(sum(len(r['graph']['nodes']) for r in selected)),str(sum(len(r['graph']['edges']) for r in selected)),' / '.join(map(str,hops))]
  assert [cell.text for cell in cells]==expected,('dataset_table',split)

 names={'bfs':'BFS','astar_admissible':'A* 下界','astar_fixed':'A* 固定','astar_ga':'A* GA','astar_ml':'A* ML','mcts':'MCTS','hybrid':'混合','hybrid_no_expert':'混合无专家'}
 for tid,split in [(8,'test'),(9,'ood')]:
  ts=doc.tables[tid]
  for cells,m in zip([r.cells for r in ts.rows[1:]],names):
   x=next(s for s in summary if s['split']==split and s['method']==m and s['budget']==8)
   expected=[names[m],f'{round(x["success_rate"]*x["n"])}/{x["n"]} ({100*x["success_rate"]:.2f}%)',f'{x["mean_expanded"]:.3f}',f'{x["mean_hops_success"]:.3f}',f'{100*x["mean_redundant_paths"]:.3f}%',f'{100*x["invalid_answer_rate"]:.2f}%']
   assert [cell.text for cell in cells]==expected,(split,m)
 for cells,x in zip([r.cells for r in doc.tables[4].rows[1:]],metrics['ga_validation']):
  expected=[str(x['seed']),f'{x["fixed_fitness"]:.6f}',f'{x["initial_champion_fitness"]:.6f}',f'{x["final_fitness"]:.6f}',f'{x["fixed_mean_expanded"]:.4f}',f'{x["final_mean_expanded"]:.4f}']
  assert [cell.text for cell in cells]==expected
 # PDF page count, text correctness, no empty overflow page, no missing glyphs.
 pdf=fitz.open(BASE/'第31组_实验三_实验报告.pdf');assert len(pdf)==structure['manual_pages'],(len(pdf),structure['manual_pages'])
 pagecheck=[];alltext=''
 for i,page in enumerate(pdf):
  t=page.get_text();alltext+=t;assert '\ufffd' not in t,('replacement_glyph',i+1);assert len(t)>120,('blank_or_overflow',i+1)
  pagecheck.append({'page':i+1,'characters':len(t),'images':len(page.get_images())})
  for b in page.get_text('dict')['blocks']:
   x0,y0,x1,y1=b['bbox'];assert x0>=50 and x1<=550 and y0>=20 and y1<=821,('pdf_bounds',i+1,b['bbox'])
 normalize=lambda t:re.sub(r'\s+','',t)
 textnorm=normalize(alltext)
 for para in doc.paragraphs:
  if para.text:assert normalize(para.text) in textnorm,('paragraph_missing_in_pdf',para.text[:80])
 for tab in doc.tables:
  for row in tab.rows:
   for cell in row.cells:assert normalize(cell.text) in textnorm,('cell_missing',cell.text)
 assert len(list((BASE/'figures').glob('*.png')))==18
 assert len(list((BASE/'figures').glob('*.svg')))==18
 badtypes={'.ppt','.pptx','.mp4','.mov','.avi','.mkv','.mp3','.wav'}
 assert not [p for p in BASE.rglob('*') if p.is_file() and p.suffix.lower() in badtypes]
 # Validate every image pair, including six Matplotlib charts outside diagram layout metadata.
 for svg in sorted((BASE/'figures').glob('*.svg')):
  root=ET.parse(svg).getroot();assert root.tag=='{http://www.w3.org/2000/svg}svg'
  vb=[float(x) for x in root.attrib['viewBox'].split()]
  with Image.open(svg.with_suffix('.png')) as image:
   assert abs((image.width/image.height)/(vb[2]/vb[3])-1)<.01,('figure_aspect',svg.name)
   image.verify()
 gallery=load(EV/'gallery_validation.json')
 assert gallery['status']=='passed' and gallery['image_count']==18 and gallery['svg_files_checked']==18
 for rel,expected in gallery['assets_sha256'].items():assert sha(BASE/rel)==expected,('gallery_audit_stale',rel)
 assert all(x['native_anchor_navigation']=='passed' and x['offline'] and x['javascript_disabled'] for x in gallery['layouts'])
 qa=BASE/'reproduce/qa'
 expected_pages={f'page_{i:02d}.png' for i in range(1,len(pdf)+1)}
 assert {p.name for p in qa.glob('page_*.png')}==expected_pages,'Stale/missing page previews'
 assert load(qa/'pdf_pages.json')['pages']==len(pdf)
 svgqa=BASE/'reproduce/qa/svg_bounds.json'
 if svgqa.exists():assert load(svgqa)==[], 'SVG text clipped; review svg_bounds.json'
 record={'status':'passed','verified_at':datetime.datetime.now().isoformat(timespec='seconds'),
  'scope':'Read-only independent data/case replay and document verification; no LLM inference or dataset regeneration.',
  'source_files_verified':source_count,'protected_project_files_unchanged':len(protected),
  'benchmark_rows_resummarized':len(rows),'summary_rows_recomputed':len(summary),'paired_intervals_recomputed':18,
  'cases_replayed':len(cases),'cases':cases,'pdf_pages':len(pdf),'docx_tables':len(doc.tables),'docx_embedded_figures':len(doc.inline_shapes),
  'image_pairs':18,'pdf_all_paragraphs_and_cells_present':True,'pdf_replacement_characters':0,'pdf_pages_checked':pagecheck,
  'no_ppt_or_video_created':True,'original_algorithms_outputs_and_ui_unchanged':True,
   'offline_gallery_verified':True,'offline_gallery_references_checked':gallery['local_references_checked'],
   'svg_files_verified':18,'pdf_preview_pages_verified':len(pdf),'team_responsibilities_verified':True,'team_member_count':5,
   'checksum_manifest_verified':CHECKSUM_FILE.exists() and not args.review_updated_files,
   'verification_script_sha256':sha(Path(__file__)),
   'verified_delivery_files_sha256':{rel:sha(path) for rel,path in inventory.items()},
  'report_files_sha256':{p.name:sha(p) for p in [BASE/'第31组_实验三_实验报告.docx',BASE/'第31组_实验三_实验报告.pdf',BASE/'第31组_实验三_实验报告.md']}}
 (EV/'delivery_validation.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in record.items() if k not in ['cases','pdf_pages_checked','report_files_sha256','verified_delivery_files_sha256','verification_script_sha256']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()