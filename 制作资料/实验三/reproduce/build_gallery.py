"""Create a self-contained offline image/case index, unrelated to the project website."""
from pathlib import Path
import json, html
from urllib.parse import quote
BASE=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
figures=read(BASE/'evidence/figure_index.json');cases=read(BASE/'evidence/case_index.json')['cases']
report_pages=read(BASE/'evidence/report_structure.json')['manual_pages']
byfig={x['name']:x for x in figures};case_map={x['case_id']:x for x in cases}
titles={'01_architecture':'混合搜索系统架构','02_ga_validation':'GA验证适应度：保留近乎平坦的真实结果','03_accuracy_budget8':'测试与OOD的有效答案率',
'04_budget_sensitivity':'扩展预算敏感性','05_cost_breakdown':'扩展、模拟及证据检查成本','06_llm_numeric_confusion':'真实LLM数值挑战混淆矩阵',
'07_negative_confirmation':'派生不可答题的正确穷尽率','08_tree_two_hop':'两跳成功路径','09_tree_three_hop':'三跳搜索完整状态树',
'10_tree_four_hop':'四跳搜索完整状态树','11_tree_mcts':'MCTS：真实树扩展与rollout','12_tree_bfs_budget':'BFS预算耗尽：生成目标不等于出队',
'13_tree_no_expert':'无专家：返回路径证据无效','14_tree_ood_ga':'OOD上GA排序退化','15_tree_no_answer':'派生不可答题：证明穷尽',
'16_tree_controlled_neutral':'受控中性评分：预算内成功','17_tree_controlled_wrong':'受控错误评分：预算耗尽','18_ood_expansion_order':'同题固定A*与GA扩展顺序'}
parts=['''<!doctype html><html lang="zh-CN" id="top"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>第31组 · 实验三图片与案例索引</title>
<style>body{max-width:1080px;margin:40px auto;padding:0 22px;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;color:#263746;background:#fff}h1,h2{line-height:1.4}a{color:#16638c}header{border-bottom:2px solid #d8e3eb;padding-bottom:20px}nav{columns:2;column-gap:38px}nav a{display:block;margin:5px 0}section{margin:44px 0;border-top:1px solid #dce5eb;padding-top:14px}img{display:block;width:auto;max-width:100%;max-height:850px;margin:16px auto;border:1px solid #e2e8ef}code{background:#edf2f6;padding:2px 5px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #d6e0e8;padding:8px;text-align:left;overflow-wrap:anywhere}th{background:#edf2f6}.note{color:#586978}.links a{margin-right:20px}@media(max-width:650px){nav{columns:1}table{font-size:12px}body{margin-top:20px}}@media print{nav{display:none}section{break-inside:avoid}img{max-height:650px}}</style></head><body>
<header><h1>实验三 · 图片与案例索引</h1><p>第31组 · 2026年11月13日</p><p>本页离线可用，不启动、不修改项目网站。图片来自冻结指标与完整搜索轨迹；点击SVG可无损放大，点击JSON可核对原始记录。</p>
<p class="links"><a href="'''+quote('第31组_实验三_实验报告.pdf')+'">阅读报告 PDF（'+str(report_pages)+'页）</a><a href="'+quote('第31组_实验三_实验报告.docx')+'">编辑 Word</a><a href="'+quote('附件说明.md')+'">附件说明</a></p></header>', '<h2>图片导航</h2><nav>']
for name,title in titles.items():parts.append(f'<a href="#{name}">{name[:2]} · {html.escape(title)}</a>')
parts.append('</nav>')
for name,title in titles.items():
 parts.append(f'<section id="{name}"><h2>{name[:2]} · {html.escape(title)}</h2>')
 parts.append(f'<p class="links"><a href="figures/{name}.png">高清PNG</a><a href="figures/{name}.svg">SVG无损放大</a></p>')
 if name in byfig:
  meta=byfig[name]
  ids=[meta['case_id']] if meta.get('case_id') else meta.get('cases',[])
  for cid in ids:
   case=case_map[cid];parts.append(f'<p><a href="{case["file"]}">{cid} · 完整JSON</a>｜{html.escape(case["method"])}｜预算{case["budget"]}｜{case["status"]}｜valid={case["valid"]}｜扩展{case["expanded"]}</p>')
  if meta.get('full_trace'):parts.append(f'<p class="note">全部{meta["states"]}个完整前缀状态，{meta["events"]}个事件；未截取子树。多事件在图中合并显示，JSON仍保留全部事件。</p>')
 if 'controlled' in name:parts.append('<p><strong>人为注入评分的机制反例，不是真实LLM预测，不用于计算模型幻觉发生率。</strong></p>')
 parts.append(f'<a href="figures/{name}.svg"><img src="figures/{name}.png" loading="lazy" alt="{html.escape(title)}"></a><p><a href="#top">返回顶部</a></p></section>')
parts.append('<section><h2>全部15个运行案例</h2><table><thead><tr><th>案例JSON</th><th>方法/预算</th><th>结果</th><th>选例用途</th></tr></thead><tbody>')
for x in cases:
 parts.append(f'<tr><td><a href="{x["file"]}">{x["case_id"]}</a></td><td>{x["method"]} / {x["budget"]}</td><td>{x["status"]}<br>valid={x["valid"]}, 扩展={x["expanded"]}</td><td>{html.escape(x["selection_reason"])}</td></tr>')
parts.append('</tbody></table><p class="note">全组统计而非个例发生率：见evidence/raw/runs.csv与summary.json。真实LLM误判及数值挑战见evidence/raw/reliability/llm_confirmation.json。</p></section></body></html>')
(BASE/'图片与案例索引.html').write_text('\n'.join(parts),encoding='utf-8')
print('Offline gallery written:',len(titles),'figures,',len(cases),'cases')