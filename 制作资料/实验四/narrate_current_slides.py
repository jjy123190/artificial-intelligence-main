"""Render checked current slides with the local built-in Mandarin SAPI voice.

Uses a standard synthetic Windows voice, not a person's cloned voice. Audio,
slide/frame identity and measurements are recorded; no internet upload required.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import wave

import imageio_ffmpeg
import pythoncom
import win32com.client

ROOT=Path(__file__).resolve().parents[2];SOURCE=ROOT/'源码';sys.path.insert(0,str(SOURCE))
from experiments.lab04.artifact_audit import audit_publication
FFMPEG=imageio_ffmpeg.get_ffmpeg_exe()


def run(args):
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    temp=Path(args.tmp).resolve();temp.mkdir(parents=True,exist_ok=True);renders=Path(args.renders).resolve()
    audit=audit_publication(Path(args.artifacts));slides=json.loads((out/'slides_evidence.json').read_text(encoding='utf-8'))
    if slides['artifact_revision']!=audit['artifact_revision']:raise ValueError('Slides and live evidence revision differ')
    overflow=json.loads((renders/'text-fit.json').read_text(encoding='utf-8'))
    if any(r['bound_width']>r['width']+3 or r['bound_height']>r['height']+2 for r in overflow):raise ValueError('Text overflows the recorded slides')
    rows={(r['method'],r['shots']):r for r in audit['core']['rows']}
    for m in ('graph','multiview','wide','wideview','continued'):rows.update({(r['method'],r['shots']):r for r in audit[m]['rows']})
    one,five=rows[('continued',1)],rows[('continued',5)]
    narration=[
      '本次实验研究少量标注条件下的新类别识别。我们使用轻量残差网络、二阶元学习和特征统计校准，连接可实际运行的网页平台。本汇报只包含已经完成并独立核验的发布结果，不把正在训练的候选当作正式成绩。',
      '小样本问题既有表征偏移，也有统计量不稳定。基础类别的判别方向未必适合新类，一张支持图片也无法直接估计方差。医疗和工业缺陷是应用动机，但本实验只在迷你图像基准验证，不声称已经完成实际业务部署。',
      '数据分为六十四个训练类、十六个验证类和二十个测试类，类别互不重叠。每类六百张图像。任务有五个类别，每类给一张或五张有标签支持图片，另有十五张查询图片。每种设置评估六百个相同配对任务，查询标签只在预测后评分。',
      '监督基线冻结编码器，以支持集均值分类。MAML使用全网络真实二阶梯度，共四千个元训练任务，测试只用支持标签进行五步适应。统计校准与基线共享初始化、样本、增强和交叉熵训练步数，区别是分布对齐正则及后续推理方法。不同训练预算和归纳、转导信息条件都单独说明。',
      '校准先借用基础训练类的均值方差，对少量支持集估计做收缩，再用无标签查询的软分配更新原型。均衡分配利用本协议每类查询数量相等的先验，不读取隐藏标签。统计先验可能与新类不匹配，原型更新也可能被错误分配带偏，因此不能预设所有模块必然有效。',
      f"在同一组测试任务上，轻量基线的一样本和五样本准确率分别为{rows[('baseline',1)]['accuracy']*100:.2f}%和{rows[('baseline',5)]['accuracy']*100:.2f}%。完整校准为{rows[('full',1)]['accuracy']*100:.2f}%和{rows[('full',5)]['accuracy']*100:.2f}%。MAML在当前预算下仍低于基线。更宽模型与额外训练带来的收益不应全部归为校准算法。",
      f"最新的同容量续训双视图达到{one['accuracy']*100:.2f}%和{five['accuracy']*100:.2f}%。参数量保持五十万九千八百五十六，但增加十六轮训练，验证选择第十二轮。相对上一版较宽双视图，配对差增加{one['paired_vs_wideview']*100:.2f}和{five['paired_vs_wideview']*100:.2f}个百分点，两个百分之九十五区间都高于零。分类配置保持不变，不能把额外训练收益说成同预算校准创新。",
      '消融结果表明，模块不是越多越好。完整方案去掉训练对齐后，两种设置的均值没有下降；仅原型更新已接近完整组合。网页可切换相对基线与模块直接对照，二者参考对象不同，不能混用显著性。原型对照同时改变均衡转导流程，不能归为单一模块作用。验证没有通过改善门槛的候选也完整保留。',
      '平均准确率不说明每个任务都稳定。轻量双视图的一样本任务中间百分之九十落在百分之三十二到七十六左右，五样本约为五十二到八十一点四。这不是均值置信区间。同容量续训保持双视图两次编码，相同结构的耗时差受到系统负载影响，不能宣传续训更快。计时在本机中央处理器上重新编码，不冒称手机或开发板实测。',
      '网站把现场推理、性能、消融、可靠性、端侧耗时和历史来源分开显示。用户可以检查误判、五类分数、真实样本和模型指纹。异常接口数据会被拒绝，而不是让页面白屏；取消等待保留已完成记录，不把服务器计算谎称为已经强制终止。配套演示录制了真实浏览器交互和实际预测。',
      '课程技术要求已对应代码和实验记录，但本地项目不能证明所有外部操作完成。组员关联、项目创建和管理员提交需要在课程平台实际核对。如果老师指定手机、ARM或开发板，需要补该设备测试。单训练种子、重复使用公开基准和均衡查询先验的限制都必须保留。',
      '核验命令会从逐任务记录重算结果，检查选择锁、权重、样本和预测。当前报告、可编辑幻灯片、真实演示和源代码对应同一个发布版本。人工智能辅助了实现和文档整理，但准确率来自实际模型运行。本实验不保证零误判或理论最优，后续优先独立数据与多训练种子验证。'
    ]
    if len(narration)!=slides['slides']:raise ValueError('Narration/slide count mismatch')
    pythoncom.CoInitialize();voice=win32com.client.Dispatch('SAPI.SpVoice');chosen=None
    for i in range(voice.GetVoices().Count):
        token=voice.GetVoices().Item(i)
        if 'Huihui' in token.GetDescription() or 'Chinese' in token.GetDescription():chosen=token;break
    if chosen is None:raise RuntimeError('No built-in Mandarin voice installed')
    voice.Voice=chosen;voice.Rate=1;voice.Volume=95
    records=[];concat=[];start=0.
    def ffmpeg(arguments):subprocess.run([FFMPEG,'-hide_banner','-loglevel','error','-y',*map(str,arguments)],check=True)
    try:
        for i,words in enumerate(narration,1):
            image=renders/f'slide-{i:02}.png'
            if not image.is_file():raise FileNotFoundError(image)
            audio=temp/f'slide-{i:02}.wav';part=temp/f'slide-{i:02}.mp4'
            stream=win32com.client.Dispatch('SAPI.SpFileStream');stream.Open(str(audio),3,False);voice.AudioOutputStream=stream
            voice.Speak(words);stream.Close()
            with wave.open(str(audio),'rb') as wav:seconds=wav.getnframes()/wav.getframerate()
            duration=seconds+.7
            ffmpeg(['-loop','1','-framerate','15','-i',image,'-i',audio,'-t',f'{duration:.3f}','-af','apad','-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-c:a','aac','-b:a','96k','-ar','48000','-movflags','+faststart',part])
            records.append({'slide':i,'narration':words,'start_seconds':start,'duration_seconds':duration,'voice':chosen.GetDescription(),
                            'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),'audio_sha256':hashlib.sha256(audio.read_bytes()).hexdigest()})
            start+=duration;concat.append("file '"+part.as_posix().replace("'","'\\''")+"'")
            print(json.dumps({'slide':i,'seconds':duration},ensure_ascii=False),flush=True)
    finally:pythoncom.CoUninitialize()
    listing=temp/'concat.txt';listing.write_text('\n'.join(concat)+'\n',encoding='utf-8')
    target=out/'实验四_当前版本PPT讲解.mp4'
    ffmpeg(['-f','concat','-safe','0','-i',listing,'-c','copy','-movflags','+faststart',target])
    evidence={'artifact_revision':audit['artifact_revision'],'voice_kind':'Local Windows standard synthetic Mandarin voice; not cloned',
              'slides_sha256':slides['slide_sha256'],'duration_seconds':start,'slides':records,'video_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    (out/'presentation_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'video':str(target),'duration_seconds':start,'slides':len(records)},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifacts',default=str(SOURCE/'outputs/lab04'))
    p.add_argument('--renders',default=str(SOURCE/'.local/current-slides-render'));p.add_argument('--tmp',default=str(SOURCE/'.local/current-narration'))
    p.add_argument('--output',default=str(ROOT/'deliverables/lab4/当前版本'));run(p.parse_args())
