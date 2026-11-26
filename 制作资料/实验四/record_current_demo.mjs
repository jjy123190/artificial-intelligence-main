/** Record the current multi-page UI and real inference; never reuse v1 selectors.
 * Emits an actual browser recording, action timeline and label-sparing inference
 * metadata. Failure leaves the prior published materials untouched.
 */
import fs from "node:fs/promises";
import path from "node:path";
import {fileURLToPath,pathToFileURL} from "node:url";
import {spawn} from "node:child_process";
import {once} from "node:events";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),"../..");
const {chromium}=await import(pathToFileURL(path.join(root,"源码/web/node_modules/playwright/index.mjs")));
const arg=(name,fallback)=>{const i=process.argv.indexOf(name);return i<0?fallback:process.argv[i+1]};
const base=arg("--base-url","http://127.0.0.1:8765");
const output=path.resolve(arg("--output",path.join(root,"deliverables/lab4/当前版本")));
const temporary=path.resolve(arg("--tmp",path.join(root,"源码/.local/current-demo-20261125")));
const ffmpeg=arg("--ffmpeg",process.env.LAB04_FFMPEG||"");
const rehearsal=process.argv.includes("--rehearse");
if(!ffmpeg&&!rehearsal)throw Error("Supply --ffmpeg executable path for video recording");
await fs.mkdir(output,{recursive:true});await fs.mkdir(temporary,{recursive:true});await fs.mkdir(path.join(temporary,"captures"),{recursive:true});
const browser=await chromium.launch({headless:true,executablePath:process.env.PLAYWRIGHT_EXECUTABLE_PATH||"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"});
const context=await browser.newContext({viewport:{width:1600,height:900},deviceScaleFactor:1,reducedMotion:"reduce"});
const page=await context.newPage();const errors=[],runs=[],steps=[];page.on("pageerror",e=>errors.push(e.message));
let recorder,recording=false,recordTask,frameCount=0,lastFrame,started=0,recorderError="";
const fps=8, videoPath=path.join(temporary,"current-demo-raw.mp4");
function elapsed(){return (Date.now()-started)/1000}
async function recordFrames(){
 while(recording){
  const before=Date.now();
  const frame=await page.screenshot({type:"jpeg",quality:75});lastFrame=frame;
  const due=Math.max(1,Math.round(elapsed()*fps)-frameCount);
  for(let i=0;i<due;i++){if(!recorder.stdin.write(frame))await once(recorder.stdin,"drain");frameCount++;}
  await new Promise(resolve=>setTimeout(resolve,Math.max(0,1000/fps-(Date.now()-before))));
 }
}
async function phase(label,fn,hold=4500){
 steps.push({label,start_seconds:elapsed()});await fn();
 await page.waitForTimeout(rehearsal?100:hold);
 steps.at(-1).end_seconds=elapsed();
}
async function show(selector){await page.locator(selector).evaluate(el=>{const y=window.scrollY+el.getBoundingClientRect().top-95;window.scrollTo(0,Math.max(0,y));});}
async function capture(name){await page.screenshot({path:path.join(temporary,"captures",name+".png")});}
async function live(compare=false){
 const wanted=compare?3:1;const responses=[];
 const handler=response=>{
  if(response.url().endsWith("/api/labs/lab04/run")&&response.request().method()==="POST")
    responses.push((async()=>({status:response.status(),ok:response.ok(),data:await response.json()}))());
 };
 page.on("response",handler);
 try{
  await page.getByRole("button",{name:compare?"同任务对比三种方法":"运行分类任务",exact:true}).click();
  await page.waitForFunction(()=>!document.querySelector(".lab4-run").disabled,null,{timeout:120000});
  const completed=await Promise.all(responses);
  const failures=completed.filter(r=>!r.ok);
  assert.deepEqual(failures,[],"A real inference request failed; do not record a partial comparison as complete");
  assert.equal(completed.length,wanted,"The browser must finish all planned methods before recording");
  const obtained=completed.map(r=>r.data);
  for(const result of obtained){assert.equal(result.compute,"live");assert.equal(result.queries.length,75);runs.push(result);}
  if(compare)for(const result of obtained)assert.deepEqual(result.queries.map(q=>q.id),obtained[0].queries.map(q=>q.id));
 }finally{page.off("response",handler)}
}

try{
 const response=await page.request.get(base+"/api/labs/lab04/snapshot");assert.equal(response.ok(),true);const snapshot=await response.json();assert.equal(snapshot.available,true);assert.equal(snapshot.verification.verified,true);assert.equal(snapshot.continued_extension.available,true);
 await page.goto(base+"/#/labs/lab04");await page.locator(".lab4-overview-paths").waitFor();await page.locator(".lab4-overview-evidence").waitFor();
 started=Date.now();
 if(!rehearsal){
  recorder=spawn(ffmpeg,["-hide_banner","-loglevel","error","-y","-f","image2pipe","-framerate",String(fps),"-vcodec","mjpeg","-i","pipe:0","-an","-c:v","libx264","-preset","veryfast","-crf","22","-pix_fmt","yuv420p","-movflags","+faststart",videoPath],{windowsHide:true,stdio:["pipe","ignore","pipe"]});
  recorder.stderr.on("data",data=>recorderError+=String(data));recorder.stdin.on("error",e=>recorderError+=e.message);recording=true;recordTask=recordFrames();
 }
 await phase("当前版本入口：方法、效果与要求分开阅读",async()=>{await capture("overview")},5000);
 await phase("现场推理：同容量续训方案，额外预算明确披露",async()=>{
  await page.locator('a[href="#/labs/lab04/inference"]').first().click();await page.locator(".lab4-setup").waitFor();
  await page.getByRole("textbox",{name:"任务种子",exact:true}).fill("20261124");
  await page.getByRole("combobox",{name:"校准策略",exact:true}).selectOption("continued");
  await page.locator(".lab4-run").scrollIntoViewIfNeeded();await capture("setup");
 },5000);
 await phase("三方法现场对照：不是预存结果回放",async()=>{await live(true);await show(".lab4-result-summary");await capture("live-result")},7000);
 await phase("单任务正确率会波动；查看支持样本与误判",async()=>{
  await show(".lab4-query-section");await page.getByRole("button",{name:/^误判/}).click();await page.locator(".lab4-query-tile").first().click();await capture("error-analysis");
 },7000);
 await phase("特征投影只解释结构，不伪造特征位移",async()=>{await show(".lab4-feature-section");await capture("features")},5500);
 await phase("相同种子切换5-shot：支持样本增加到25张",async()=>{
  await page.locator(".lab4-shot-cards button").nth(1).click();await page.locator(".lab4-run").scrollIntoViewIfNeeded();await live(false);await show(".lab4-result-summary");await capture("five-shot");
 },6500);
 await phase("性能页：600个任务均值与95%区间，不是演示分数",async()=>{
  await page.getByRole("link",{name:"对比与消融",exact:true}).click();await page.getByRole("combobox",{name:"重点查看",exact:true}).selectOption("continued");await page.getByRole("button",{name:"5-shot",exact:true}).click();await show(".lab4-accuracy-plot");await capture("performance");
 },7000);
 await phase("独立消融页：指定模块直接对照与Baseline口径",async()=>{await page.getByRole("link",{name:"模块消融",exact:true}).click();await page.getByRole("button",{name:"模块直接对照",exact:true}).click();await show(".lab4-paired-plot");await capture("ablations")},6500);
 await phase("可靠性页：概率分数不等于正确率保证",async()=>{await page.getByRole("link",{name:"可靠性",exact:true}).click();await show(".lab4-calibration-plot");await capture("reliability")},6500);
 await phase("端侧性能页：双视图增加编码代价",async()=>{await page.getByRole("link",{name:"端侧性能",exact:true}).click();await show(".lab4-latency-plot");await capture("deployment")},6500);
 await phase("要求与交付：本机证据与平台提交动作分开",async()=>{await page.getByRole("link",{name:"要求与交付",exact:true}).click();await show(".lab4-requirement-list");await capture("requirements")},6500);
 await phase("课程平台与指定硬件仍需实际核对",async()=>{await show(".lab4-requirement-boundary");await capture("boundaries")},5000);
 assert.deepEqual(errors,[]);assert.equal(runs.length,4);assert.ok(runs.every(r=>r.artifact_revision===snapshot.provenance.artifact_revision));
 if(!rehearsal){
  recording=false;await recordTask;recorder.stdin.end();const [code]=await once(recorder,"exit");if(code!==0)throw Error(recorderError||`ffmpeg exited ${code}`);
  const target=path.join(output,"实验四_当前页面与真实推理_Demo.mp4");await fs.copyFile(videoPath,target);
  function stamp(seconds){const ms=Math.max(0,Math.round(seconds*1000));return `${String(Math.floor(ms/3600000)).padStart(2,"0")}:${String(Math.floor(ms/60000)%60).padStart(2,"0")}:${String(Math.floor(ms/1000)%60).padStart(2,"0")},${String(ms%1000).padStart(3,"0")}`;}
  const subtitles=steps.map((step,i)=>`${i+1}\n${stamp(step.start_seconds)} --> ${stamp(step.end_seconds)}\n${step.label}\n`).join("\n");
  await fs.writeFile(path.join(output,"实验四_当前页面与真实推理_Demo.srt"),subtitles,"utf8");
 }
 const result={recorded:!rehearsal,kind:"actual interactive browser recording; no audio",artifact_revision:snapshot.provenance.artifact_revision,viewport:{width:1600,height:900},fps,frames:frameCount,duration_seconds:frameCount/fps,steps,runs:runs.map(r=>({method:r.method,calibration:r.calibration,seed:r.seed,shots:r.shots,correct:r.correct,total:r.total,accuracy:r.accuracy,checkpoint_sha256:r.checkpoint_sha256,artifact_revision:r.artifact_revision,support_ids:r.support.map(s=>s.id),query_ids:r.queries.map(q=>q.id)})),console_errors:errors};
 if(!rehearsal)result.video_sha256=createHash("sha256").update(await fs.readFile(path.join(output,"实验四_当前页面与真实推理_Demo.mp4"))).digest("hex");
 await fs.writeFile(path.join(rehearsal?temporary:output,"demo_evidence.json"),JSON.stringify(result,null,2),"utf8");console.log(JSON.stringify({recorded:!rehearsal,steps:steps.length,runs:runs.length,frames:frameCount,duration_seconds:frameCount/fps,output}));
}finally{
 recording=false;if(recordTask)await recordTask.catch(()=>{});if(recorder&&!recorder.stdin.destroyed)recorder.stdin.end();await context.close();await browser.close();
}
