"""Course navigation and explicit, allowlisted local deliverables."""
from __future__ import annotations
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LABS={
 "lab01":{"id":"lab01","number":"01","name":"知识表示、知识增强与推理","summary":"用同一份股权知识比较不同表示，演示规则和图查询，再观察知识增强。","code":"experiments/lab01/","entry":"#/labs/lab01","modules":[
  {"id":"representation","title":"知识表示与方法比较","requirement":"核心任务（1）：至少两种主流知识表示方法，比较优缺点和适用场景。","description":"图谱三元组、产生式规则与框架继承；关系表作为同知识查询对照。","kind":"核心要求"},
  {"id":"industry","title":"行业案例与落地流程","requirement":"核心任务（2）：结合真实行业案例说明知识表示与推理的落地流程。","description":"真实行业背景与虚构演示数据明确分开，展示数据到知识再到人工复核的流程。","kind":"核心要求"},
  {"id":"demo","title":"知识增强与推理演示","requirement":"核心任务（3）：独立完成小型规则推理或知识图谱路径查询。","description":"运行原实验一规则、股权简单路径穿透和最短关联路径，查看新增知识与来源。","kind":"核心要求"},
  {"id":"limitations","title":"局限与发展方向","requirement":"核心任务（4）：客观分析现存局限和未来发展方向。","description":"区分控制与经济权益、规则假设与事实、简单路径与无限循环。","kind":"核心要求"}]},
 "lab02":{"id":"lab02","number":"02","name":"确定性、不确定性与混合推理","summary":"从可证明的命题到带不确定性的评分，量化比较算法优化与混合策略。","code":"experiments/lab02/","entry":"#/labs/lab02","modules":[
  {"id":"modeling","title":"知识表示与建模","requirement":"基础层：同一复杂应用场景与一阶谓词逻辑知识库。","description":"查看案例、谓词事实、27条结构规则和可追溯的数据映射。","kind":"基础要求"},
  {"id":"deterministic","title":"确定性推理","requirement":"基础层：至少一种确定性推理，输出完整推理路径与证明步骤。","description":"运行正向／表格化反向查询，并查看独立股权子案例的归结证明。","kind":"基础要求"},
  {"id":"uncertain","title":"不确定性推理","requirement":"基础层：至少两种不确定性方法，比较算法特点与参数。","description":"对同一企业指标运行Bayes与Mamdani，查看参数、评分、规则激活及缺测语义。","kind":"基础要求"},
  {"id":"optimization","title":"算法优化与对比实验","requirement":"进阶层：优化推理方法，多组量化对比运行时间、准确性、容错性与图表。","description":"原始／优化、训练／验证／测试、噪声与缺测结果；逐项读实测而不是手填指标。","kind":"进阶要求"},
  {"id":"hybrid","title":"混合推理与创新验证","requirement":"创新层：本项目选择确定性＋不确定性混合推理并验证效果。","description":"对比不锁定、结构锁定、校准复核；分开解释分数、证明与最终行动。","kind":"创新要求"}]} }
FILES={
 "lab01":{
 "report":("课堂报告", "deliverables/lab01/课堂报告.docx","report"),
 "slides":("课堂PPT", "deliverables/lab01/企业股权知识表示与风险推理.pptx","slides"),
 "presentation-video":("PPT汇报视频", "deliverables/lab01/实验一_PPT汇报.mp4","video"),
 "demo-video":("网页版Demo", "deliverables/lab01/实验一_Demo.mp4","video")},
 "lab02":{
 "report":("正式课程报告", "deliverables/lab02/第31组_人工智能推理实验报告.docx","report"),
 "slides":("课堂PPT", "deliverables/lab02/推理实验汇报.pptx","slides"),
 "innovation":("创新点说明", "deliverables/lab02/第31组_创新点说明.docx","report"),
 "presentation-video":("PPT汇报视频", "deliverables/lab02/实验二_PPT汇报.mp4","video"),
 "demo-video":("网页版Demo", "deliverables/lab02/实验二_Demo.mp4","video")}}

def catalog():
    labs=[]
    for lab in LABS.values():
        files=[]
        for id,(label,relative,kind) in FILES[lab["id"]].items():
            path=ROOT/relative
            # Experiment 1 is a retained-file inventory, not a to-do list.
            # Legacy file metadata/downloads remain compatible; the website has no materials view.
            if lab["id"] == "lab01" and not path.is_file():
                continue
            files.append({"id":id,"label":label,"kind":kind,"path":relative,"available":path.is_file(),"bytes":path.stat().st_size if path.is_file() else 0,
                "href":"/api/course/files/"+lab["id"]+"/"+id})
        labs.append({**lab,"files":files})
    return {"title":"股权知识与智能推理实验平台","scene":"企业股权关系、结构线索与人工复核","labs":labs,"note":"网站仅展示实验、交互与可视化，共享案例与基础能力；各实验方法和结果独立版本。"}

def file_path(lab,identifier):
    if lab not in FILES or identifier not in FILES[lab]:raise KeyError("文件入口不存在")
    relative=FILES[lab][identifier][1];path=(ROOT/relative).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():raise FileNotFoundError("材料尚未生成或不可用")
    return path
