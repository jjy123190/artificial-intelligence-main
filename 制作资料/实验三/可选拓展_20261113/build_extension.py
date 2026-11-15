"""Insert the migration supplement into the saved user DOCX snapshot.
Only word/document.xml is changed; all other ZIP entries stay byte-for-byte equal.
Never writes to the formal deliverables directory or an open user document.
"""
from pathlib import Path
from copy import deepcopy
from zipfile import ZipFile
from hashlib import sha256
import json
from lxml import etree
from docx import Document
from docx.shared import Cm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

BASE = Path(__file__).resolve().parent
NAME = '第31组_实验三_实验报告.docx'
SOURCE = BASE / 'before' / NAME
DEST = BASE / 'draft' / NAME
RESULT = json.loads((BASE / 'example_verification.json').read_text(encoding='utf-8'))
assert RESULT['positive_examples_accepted'] == 2 and RESULT['negative_cases_rejected'] == 8

CONTENT = []

def block(kind, text=None, **kwargs):
    row = {'kind': kind, **kwargs}
    if text is not None: row['text'] = text
    CONTENT.append(row)

block('h1', '14 可选拓展：数学定理证明与规划迁移', new_page=True)
block('p', '本节把企业关系图上的“候选生成—启发排序—规则核验”抽象为跨领域方案，并给出两个可逐步复核的自构造样例。已完成的是迁移设计和给定证书/计划的核验，不是新领域的完整搜索实现；下述结果不并入原3,888次主实验。')
block('h2', '14.1 具体迁移方案与接口边界')
block('p', '当前search.py显式限定2—4跳，g取路径深度，MCTS模拟也以query.hops为终点；domain.py的关系序列、实体防环与特征均服务于企业图谱。因此可复用搜索策略、GA优化流程和日志评价思路，但不能只换数据就直接用于证明或规划。首先应抽离表23所列领域接口。')
block('table', caption='表23 迁移适配层的核心接口（设计方案）', widths=[3.6, 6.4, 6.4],
      headers=['接口/职责', '数学定理证明', '规划任务'], rows=[
          ['初态与状态键\ninitial_state / state_key', '保存上下文Γ、全部未闭合目标及证明孔/约束；去重必须保留这些语义，不能只看一个目标。', '保存规范化世界事实与计划前缀；去重按世界状态及必要资源，以best_g保留低成本到达。'],
          ['合法动作与转移\nlegal_actions / transition', '规则经类型/前提检查后展开；选规则是OR，规则产生的多个子目标是AND，必须全部关闭。', '枚举满足前置条件的动作，按增加/删除效应更新事实；不能沿用“路径内实体不重复”。'],
          ['终止与证书核验\ngoal_test / verify_certificate', '未闭合目标为空只是候选完成；还须检查完整证明证书，拒绝未定义引理或证明孔。', '从初态重放全部动作，核查每一步和最终目标；单个合法动作不等于计划完成。'],
          ['代价与启发特征\nstep_cost / features', '预先约定推理步或工具成本；特征可含目标结构、可用引理匹配和未闭合目标数。', 'g累加真实动作代价；特征可含未满足目标、约束冲突、资源余量与松弛问题估值。'],
      ], font=9.5)
block('p', '搜索层需取消固定跳数终止。BFS仅在等代价动作下作为最少步数基线；A*若使用不可采纳的学习估值，只能视为启发排序，不能承诺最优。规划状态应按best_g更新，发现更低代价时允许重开；证明则要维护完整目标集。MCTS应以证书/目标核验或预算上限结束模拟，截断返回unknown，不能当作命题为假或任务无解。')
block('p', '优化层需重建领域特征和训练标签：已核验证书或可执行计划提供成功样本，预算内未找到解不自动标为失败真值；GA权重须在新领域验证集重新选择。LLM仍只评估已生成候选的合法性与价值，并解释已核验结果，不能直接提交答案。模型软评分可以改变优先级，不能绕过硬规则或证书检查器。')

block('h2', '14.2 样例一：等式定理的证明义务与证书', new_page=True)
block('p', '设a、b、c属于同一类型，已知Γ={h1: a=b，h2: b=c}，目标为Γ ⊢ c=a。此例仅使用假设引用、等式对称与传递：由x=y得到y=x；由x=y和y=z得到x=z。两条等式规则与Lean官方文档中的Eq.symm、Eq.trans一致[6]，但本轮没有运行Lean。')
block('p', '将状态记为S=(Γ，待证目标列表，证明构造记录)，每次处理最左侧目标。对称操作把c=a改写为待证a=c；选择中间项b进行传递，则生成a=b和b=c两个必须同时完成的子目标。表24给出了全部逆向步骤，不能在第3步只关闭一个分支时提前宣布成功。')
proof_actions = ['初始状态', '对称：Eq.symm', '传递：中间项b', '引用假设：exact h1', '引用假设：exact h2']
proof_notes = ['目标未完成', '保留外层对称证书孔', '生成两个AND子目标', '只关闭a=b，仍欠b=c', '无剩余目标，再核验完整证书']
proof_rows=[]
for row in RESULT['theorem']['trace']:
    goals='；'.join(a+'='+b for a,b in row['open_goals']) or '∅（全部关闭）'
    proof_rows.append([str(row['step']), proof_actions[row['step']], goals, proof_notes[row['step']]])
block('table', caption='表24 等式定理的逐步证明义务（已逐步核验）', widths=[1.0, 4.2, 4.2, 7.0],
      headers=['步', '所选动作', '动作后的未闭合目标', '核验要点'], rows=proof_rows, font=9.5)
block('p', '最终证书可写为：', indent=False, keep_next=True)
block('code', 'p = Eq.symm (Eq.trans h1 h2)')
block('p', '顺向复核时，h1与h2先经传递得到a=c，再经对称得到c=a。独立脚本从叶节点读取真实假设，自底向上推导结论，而不是信任证书自称“正确”。该证书有2个非平凡等式推理节点和2个假设叶节点；对应表24的4次逆向动作。这些是给定推演的规模，不是搜索扩展数或性能提升。')
block('p', '错误分支也能定位：直接组合Eq.trans h2 h1时，中间项c与a不一致；把h2替换为不存在的h3时，引用检查失败；删除传递规则的右子证书时，结构检查失败；只执行到第3步时，仍有b=c未关闭。四种情况均被脚本拒绝，分别覆盖方向不匹配、虚构前提、遗漏AND分支和未完成证明。')
block('p', '与完整迁移的差距在于，本例只核验有限符号上的给定等式证书，不自动发现中间引理，也不支持量词、归纳或依赖类型。实际系统应把候选步骤交给可信证明内核并保存可重放证书；一次受限检查通过不能代替通用证明器的认证。')

block('h2', '14.3 样例二：三积木规划的动作与状态', new_page=True)
block('p', '采用STRIPS式动作表示[7]，构造单机械臂、完全可观测且确定性的三积木任务。记T(x)=在桌面，O(x,y)=x在y上，C(x)=顶部空闲，H(x)=被夹持，E=手空。未列出的事实为假；被夹持积木不记C；二元动作要求x≠y。每步代价为1，执行前须有Pre⊆S，再按S′=(S−Del)∪Add更新。')
block('table', caption='表25 积木动作的前置条件与效应', widths=[3.2, 4.4, 4.4, 4.4],
      headers=['动作', 'Pre：执行前必须成立', 'Add：增加事实', 'Del：删除事实'], rows=[
          ['PickUp(x)', 'T(x), C(x), E', 'H(x)', 'T(x), C(x), E'],
          ['PutDown(x)', 'H(x)', 'T(x), C(x), E', 'H(x)'],
          ['Stack(x,y)', 'H(x), C(y)', 'O(x,y), C(x), E', 'H(x), C(y)'],
          ['Unstack(x,y)', 'O(x,y), C(x), E', 'H(x), C(y)', 'O(x,y), C(x), E'],
      ], font=9.3)
block('p', '初态S₀={T(A), T(B), T(C), C(A), C(B), C(C), E}，目标G={O(A,B), O(B,C), E}。给定计划先把B放到C上，再把A放到B上；表26列出每步后的完整事实集合，核验器同时检查单夹持、唯一支撑、无支撑环和Clear一致性。')
short_names={'ontable':'T','on':'O','clear':'C','holding':'H','handempty':'E'}
def compact_fact(fact):
    head=fact.split('(',1)[0]
    return short_names[head]+(fact[len(head):] if '(' in fact else '')
plan_rows=[]
for row in RESULT['planning']['trace'][1:]:
    plan_rows.append([str(row['step']),row['action'], '{'+', '.join(compact_fact(x) for x in row['state'])+'}',str(row['g'])])
block('table', caption='表26 给定计划的完整状态重放（全部动作合法）', widths=[1.0, 3.5, 10.8, 1.1],
      headers=['步', '动作', '执行后的世界事实S', 'g'], rows=plan_rows, font=9.4)
block('p', '第4步满足G，计划长度和总代价均为4。此例还能手工核对最优性：两个目标O事实初始均不存在，每个只能由一次Stack产生；初始手空，每次Stack又消耗Holding，因此至少需要2次Stack和2次获取积木的动作。下界为4，给定计划恰好达到；这只对表25的单位代价模型成立，并非运行迁移A*得到的最优性结果。')
block('p', '反向检查中，初态直接Stack(A,B)因缺H(A)被拒；PickUp(B)后再PickUp(C)因手不空被拒；只执行前两步虽均合法，却因缺O(A,B)未达目标；Stack(A,A)因参数相同被拒。四类错误说明，动作合法、状态一致与完整目标达成应分开检查，LLM给出的高分不能覆盖任何一项失败。')

block('h2', '14.4 验证指标、核验结果与局限', new_page=True)
block('p', '完整迁移后的评价应沿用“固定任务—明确预算—独立核验—保存轨迹”的原则。先建立无学习基线，再分别加入GA、路径价值学习、LLM评分与专家约束做消融；BFS用于等代价任务，A*与MCTS按各自机制记录额外开销。相同扩展预算并不代表相同总成本，表27的指标应联合报告。以下为待实施协议，不是已取得的新领域实验成绩。')
block('table', caption='表27 跨领域评估指标与记录口径（待实施）', widths=[3.0, 8.3, 5.1],
      headers=['评价维度', '建议记录的指标', '判读约束'], rows=[
          ['可核验完成率', '通过证书或完整计划核验的任务数/全部预先固定任务数；并列记录unknown、非法结果和工具错误。', '预算超限不从分母剔除；unknown不等于命题假或已证明无解。'],
          ['搜索与工具成本', '状态扩展、候选生成、前提/证书检查、MCTS模拟、模型调用与token、墙钟耗时及峰值内存。', '固定预算外另报实际成本；不把少扩展直接写成更快。'],
          ['解的质量', '证明节点/约定推理步；计划长度与真实总代价；对已知最优的小任务记录代价差距。', '不可采纳启发不保证最优；不同证明规则的步数不可直接混比。'],
          ['约束与安全性', '被拒候选数/送检候选数、重复状态率；在标注反例上统计误接受和误拒绝。', '高拒绝率也可能是候选质量差；拒绝率不是准确率。'],
          ['泛化与稳定性', '按定理结构/依赖族或规划任务族隔离训练、验证、测试；另测对象规模与目标组合变化。', '先去重再划分；GA/随机搜索多种子，报告波动及失败轨迹。'],
      ], font=9.5)
block('p', '本轮实际核验：verify_examples.py接受了1份等式证书与1条四步计划，并拒绝4个证明错误分支、4个规划错误分支，共2个正例和8个负向检查。两例均保留完整中间状态、拒绝原因和JSON结果；脚本不依赖LLM、不训练模型，也未调用现有lab03搜索。这组检查只说明样例在所声明规则下可复核，不能估计泛化准确率或宣称GA/LLM已带来迁移收益。')
block('p', '局限首先来自任务规模和验证器范围：等式例没有量词、归纳或引理检索，Python受限检查器也不等于Lean等可信内核；三积木例不含部分可观测、随机效应、时间约束和并发动作。其次，少量人工构造反例只覆盖已列缺陷，不能证明验证器普遍无漏洞。学习启发还可能把预算引向错误分支，训练截断样本与测试泄漏需单独控制。')
block('p', '后续实现应先完成领域适配器与可信验证器接入，再在隔离任务集上比较相同预算下的可核验完成率和总成本。若只改善排序而未改善证书通过率，结论应如实保留。两类迁移共同保留的核心原则是：学习模块负责建议搜索顺序，领域规则负责约束状态转移，最终正确性由可重放证据决定。')
block('note', '复核入口：配套目录lab3_备份/可选拓展_20261113中的verify_examples.py与example_verification.json；运行方法及最新版报告检查见附录A。')

PARAGRAPH_UPDATES = {
    16: '报告同时展示2、3、4跳与MCTS真实搜索树，并用预算耗尽、无专家无效路径、OOD排序退化、真实LLM误判和受控错误评分分析失效机制。全部图表可追溯到附件JSON。主实验仅汇编冻结结果和重放典型案例，没有重新训练、改变基准问题或增加LLM调用。第14章另给出跨领域迁移设计与两个自构造核验样例。',
    19: '阅读提示：第7—9章看总体结果，第10—11章用轨迹解释结果，第14章为可选迁移设计与样例核验。源码、训练/验证来源、原始模型输出和受控注入相互区分。正文的evidence/、figures/、reproduce/均相对配套deliverables/lab3_备份目录。',
    190: '可选拓展见第14章：分别给出数学定理证明与规划的领域接口、可逐步核验样例、拟定指标及局限。迁移需要重定义状态、动作、终止与成本；本轮只完成方案和样例核验，未把原框架直接改造成通用证明器或规划器。',
    198: '外部文献解释经典方法与相关工作，不作为项目指标来源。原[1]—[5]访问记录见evidence/reference_checks.json；新增[6]—[7]的官方文档与Crossref元数据核验见可选拓展_20261113/reference_checks.json。',
    199: '表28 报告中主实验结论的本地证据定位',
    200: '表28路径以deliverables/lab3_备份/evidence为根，raw文件逐字节保留。模型权重清单见原项目data/lab03/llm_model_manifest.json；原保护文件指纹见备份内reproduce/qa/protected_before.json。第14章样例及新版报告验收另存于可选拓展_20261113，不与原26页报告的验收清单混用。',
    203: '在项目根目录打开PowerShell。下列命令只核验给定样例、最新版报告清单与原证据指纹，不重建主实验、不训练、不调用LLM；verify_report.py不等同于重新执行完整算法验收。',
    204: "Set-Location 'deliverables/lab3_备份/可选拓展_20261113'\npython -B -X utf8 verify_examples.py\npython -B -X utf8 verify_report.py\nSet-Location ../../..",
    207: '本次未执行上述重建命令。重跑算法后应整套更新证据，不能混用新模型/权重与旧指标。备份内旧reproduce/build_report.py仅对应原版报告；本次增补源和核验材料在可选拓展_20261113。不要用旧生成器或lab03 report/all入口覆盖当前报告。',
    209: '表29 重点附件定位',
    210: '表29中figures/与evidence/均相对lab3_备份，cases/指evidence/cases/，完整文件名见其中的图片与案例索引.html。F09仅扩展根1次、证明剪枝6个候选后返回exhausted，oracle为0条解；它使用图中不存在的目标行业，不同于F01的budget_exhausted，也不计入可答题准确率。',
    211: '正式lab3目录只保留报告DOCX/PDF，图片、冻结证据及复核脚本归档在lab3_备份。本次未生成PPT或视频。成员职责见附录B；课程平台项目、组员关联与管理员提交仍由小组自行确认。',
    215: '表30 成员分工与协作安排',
}
REFERENCES = [
    '[6] The Lean Community. Theorem Proving in Lean 4: Quantifiers and Equality[EB/OL]. https://lean-lang.org/theorem_proving_in_lean4/Quantifiers-and-Equality/，访问日期：2026-11-13（等式对称与传递规则）。',
    '[7] Fikes R E, Nilsson N J. STRIPS: A New Approach to the Application of Theorem Proving to Problem Solving[J]. Artificial Intelligence, 1971, 2(3–4): 189–208. DOI: 10.1016/0004-3702(71)90010-5.',
]
NAV_ROW = ['可选拓展：定理证明与规划', '第14章', '迁移接口；2个样例；正反核验']


def replace_paragraph(p, text):
    if p.runs:
        p.runs[0].text = text
        for run in list(p.runs[1:]): run._element.getparent().remove(run._element)
    else: p.add_run(text)


def set_fonts(run, size=11, bold=None, family='Times New Roman'):
    run.font.name = family
    run.font.size = Pt(size)
    if bold is not None: run.font.bold = bold
    pr = run._element.get_or_add_rPr()
    fonts = pr.find(qn('w:rFonts'))
    if fonts is None:
        fonts = OxmlElement('w:rFonts'); pr.insert(0, fonts)
    for k, v in [('ascii', family), ('hAnsi', family), ('eastAsia', '宋体')]: fonts.set(qn('w:'+k), v)
    # Explicit fonts are used only on new prose/cells, not on original user styles.
    for k in ['asciiTheme','hAnsiTheme','eastAsiaTheme']:
        fonts.attrib.pop(qn('w:'+k),None)


def add_paragraph(anchor, text, kind='p', new_page=False, indent=True, keep_next=False):
    style={'h1':'Heading 1','h2':'Heading 2','caption':'Caption'}.get(kind,'Normal')
    p=anchor.insert_paragraph_before(text,style)
    pf=p.paragraph_format
    pf.page_break_before=bool(new_page)
    pf.widow_control=True
    if kind in {'h1','h2'}:
        pf.keep_with_next=True;pf.keep_together=True
        for run in p.runs:
            color=OxmlElement('w:color');color.set(qn('w:val'),'auto');run._r.get_or_add_rPr().append(color)
    else:
        pf.keep_with_next=keep_next or kind=='caption'
        pf.space_before=Pt(0);pf.space_after=Pt(5)
        pf.line_spacing=1.24 if kind=='p' else 1.15
        pf.first_line_indent=Pt(22) if kind=='p' and indent else Pt(0)
        p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY if kind=='p' else WD_ALIGN_PARAGRAPH.LEFT
        size=11 if kind=='p' else 9.5
        if kind=='code':size=10
        for run in p.runs:set_fonts(run,size, family='Consolas' if kind=='code' else 'Times New Roman')
        if kind=='caption':
            for run in p.runs:
                color=OxmlElement('w:color');color.set(qn('w:val'),'auto');run._r.get_or_add_rPr().append(color)
        if kind=='code':
            pf.keep_together=True
            shd=OxmlElement('w:shd');shd.set(qn('w:fill'),'EFF5F8');p._p.get_or_add_pPr().append(shd)
    return p


def add_table(doc, anchor, item, template):
    add_paragraph(anchor,item['caption'],'caption')
    table=doc.add_table(rows=1, cols=len(item['headers']))
    table.alignment=WD_TABLE_ALIGNMENT.CENTER
    table.autofit=False
    # Use the saved user's existing table style, borders and palette.
    old=table._tbl.tblPr
    table._tbl.replace(old,deepcopy(template._tbl.tblPr))
    pr=table._tbl.tblPr
    tw=pr.find(qn('w:tblW'));tw.set(qn('w:type'),'dxa');tw.set(qn('w:w'),str(round(sum(item['widths'])*567)))
    for col,width in zip(table.columns,item['widths']): col.width=Cm(width)
    for row_values in item['rows']: table.add_row()
    for ri,(row,values) in enumerate(zip(table.rows,[item['headers']]+item['rows'])):
        trpr=row._tr.get_or_add_trPr()
        cant=OxmlElement('w:cantSplit');trpr.append(cant)
        if ri==0:
            repeat=OxmlElement('w:tblHeader');trpr.append(repeat)
        for ci,(cell,text,width) in enumerate(zip(row.cells,values,item['widths'])):
            cell.width=Cm(width);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cell.text=text
            tcpr=cell._tc.get_or_add_tcPr()
            margins=OxmlElement('w:tcMar')
            for side,value in [('top',65),('bottom',65),('left',80),('right',80)]:
                node=OxmlElement('w:'+side);node.set(qn('w:w'),str(value));node.set(qn('w:type'),'dxa');margins.append(node)
            for x in list(tcpr.findall(qn('w:tcMar'))):tcpr.remove(x)
            tcpr.append(margins)
            shd=OxmlElement('w:shd');shd.set(qn('w:fill'),'E8EFF5' if ri==0 else ('F5F7FA' if ri%2==0 else 'FFFFFF'));tcpr.append(shd)
            for p in cell.paragraphs:
                pf=p.paragraph_format;pf.space_before=Pt(0);pf.space_after=Pt(0)
                pf.line_spacing=1.12;pf.first_line_indent=Pt(0);pf.keep_together=True
                pf.keep_with_next=False
                for run in p.runs:set_fonts(run,item.get('font',9.5),bold=(ri==0))
    table._tbl.getparent().remove(table._tbl)
    anchor._p.addprevious(table._tbl)
    return table


def main():
    doc=Document(SOURCE)
    original_paragraphs=list(doc.paragraphs)
    original_tables=list(doc.tables)
    assert len(original_paragraphs)==219 and len(original_tables)==25
    assert original_paragraphs[192].text=='参考文献与证据来源'
    for index,text in PARAGRAPH_UPDATES.items():replace_paragraph(original_paragraphs[index],text)
    nav=deepcopy(original_tables[0].rows[-1]._tr)
    for element in nav.iter():
        element.attrib.pop(qn('w14:paraId'),None)
        element.attrib.pop(qn('w14:textId'),None)
    original_tables[0]._tbl.append(nav)
    for cell,text in zip(original_tables[0].rows[-1].cells,NAV_ROW):
        replace_paragraph(cell.paragraphs[0],text)
    anchor=original_paragraphs[192]
    for item in CONTENT:
        if item['kind']=='table':add_table(doc,anchor,item,original_tables[0])
        else:add_paragraph(anchor,item['text'],item['kind'],item.get('new_page',False),item.get('indent',True),item.get('keep_next',False))
    for text in REFERENCES:
        p=original_paragraphs[198].insert_paragraph_before(text,'Normal')
        p._p.insert(0,deepcopy(original_paragraphs[197]._p.pPr)) if original_paragraphs[197]._p.pPr is not None else None
        for run in p.runs:set_fonts(run,9.5)
    doc_xml=etree.tostring(doc._element,encoding='UTF-8',xml_declaration=True,standalone=True)
    DEST.parent.mkdir(exist_ok=True)
    with ZipFile(SOURCE) as src, ZipFile(DEST,'w') as dst:
        for info in src.infolist(): dst.writestr(info,doc_xml if info.filename=='word/document.xml' else src.read(info.filename))
    out=Document(DEST)
    after_paragraphs=doc.paragraphs
    locations={id(p._p):i for i,p in enumerate(after_paragraphs)}
    spec={
        'base_sha256':sha256(SOURCE.read_bytes()).hexdigest(),
        'paragraph_updates':{str(k):v for k,v in PARAGRAPH_UPDATES.items()},
        'original_paragraph_after_index':{str(i):locations[id(p._p)] for i,p in enumerate(original_paragraphs)},
        'nav_row_added':NAV_ROW,'new_tables':5,'expected_table_count':30,
        'references_added':REFERENCES,'expected_media_count':14,
        'scope':'Only document.xml changed; all original text and tables preserved except the listed navigation/path/caption updates.'
    }
    (BASE/'qa/update_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    (BASE/'extension_content.json').write_text(json.dumps(CONTENT,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['<!-- 本文件仅为新增第14章的可编辑内容，不是原26页报告的替代全文。 -->','']
    for item in CONTENT:
        kind=item['kind']
        if kind=='table':
            lines += [item['caption'],'','| '+' | '.join(item['headers'])+' |','| '+' | '.join('---' for _ in item['headers'])+' |']
            for r in item['rows']:lines.append('| '+' | '.join(str(x).replace('\n','<br>') for x in r)+' |')
        elif kind=='code':lines += ['```text',item['text'],'```']
        else:lines.append(('# ' if kind=='h1' else '## ' if kind=='h2' else '')+item['text'])
        lines.append('')
    lines += ['## 本节新增参考文献','',*REFERENCES,'']
    (BASE/'第14章_可选迁移拓展.md').write_text('\n'.join(lines),encoding='utf-8')
    print('DRAFT',DEST)
    print('PARAGRAPHS',len(out.paragraphs),'TABLES',len(out.tables),'INLINE_IMAGES',len(out.inline_shapes))
    with ZipFile(SOURCE) as z1,ZipFile(DEST) as z2:
        changes=[x for x in z1.namelist() if z1.read(x)!=z2.read(x)]
        assert changes==['word/document.xml'],changes
        print('CHANGED_PACKAGE_PARTS',changes)


if __name__=='__main__':main()
