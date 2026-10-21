const INFO={
 Healthy:{t:"Healthy leaf",e:"🍃",c:"var(--healthy)",d:"No sign of powdery mildew or rust was detected. Keep monitoring, water at the base of the plant and keep good airflow around it."},
 Powdery:{t:"Powdery mildew",e:"❄️",c:"var(--powdery)",d:"A fungal disease that shows as a white, powdery coating on leaves. Remove badly affected leaves, improve airflow, avoid wetting the foliage and ask a garden centre about a suitable fungicide."},
 Rust:{t:"Leaf rust",e:"🍂",c:"var(--rust)",d:"A fungal disease that shows as orange or brown pustules, often under the leaf. Remove infected leaves, avoid overhead watering and consider a suitable fungicide if it spreads."}
};
const $=id=>document.getElementById(id);
let file=null;
fetch("/health").then(r=>r.json()).then(h=>{
  $("dot").className="dot "+(h.model_loaded?"ok":"bad");
  $("status").textContent=h.model_loaded?"Model online · "+h.model_name:"Model not loaded";
}).catch(()=>{$("dot").className="dot bad";$("status").textContent="API offline"});

function pick(f){
  if(!f||!f.type.startsWith("image/"))return;
  file=f;$("preview").src=URL.createObjectURL(f);
  $("drop").classList.add("has");$("hint").hidden=true;
  $("go").disabled=false;$("reset").hidden=false;
}
$("drop").onclick=()=>$("file").click();
$("file").onchange=e=>pick(e.target.files[0]);
$("reset").onclick=()=>$("file").click();
["dragover","dragenter"].forEach(ev=>$("drop").addEventListener(ev,e=>{e.preventDefault();$("drop").classList.add("over")}));
["dragleave","drop"].forEach(ev=>$("drop").addEventListener(ev,e=>{e.preventDefault();$("drop").classList.remove("over")}));
$("drop").addEventListener("drop",e=>pick(e.dataTransfer.files[0]));

$("go").onclick=async()=>{
  $("go").disabled=true;
  $("out").innerHTML='<div class="empty"><div class="spin"></div><p>Analyzing the leaf…</p></div>';
  const fd=new FormData();fd.append("file",file);
  try{
    const r=await fetch("/predict",{method:"POST",body:fd});
    const j=await r.json();
    if(!r.ok)throw new Error(j.detail||"Request failed");
    show(j);
  }catch(err){
    $("out").innerHTML='<div class="empty fade"><div>⚠️</div><p>'+(err.message==="Failed to fetch"?"Could not reach the API. Is the server running?":err.message)+'</p></div>';
  }
  $("go").disabled=false;
};

function show(j){
  const k=INFO[j.label]||{t:j.label,e:"🌱",c:"var(--brand)",d:""};
  const pct=v=>(v*100).toFixed(1)+"%";
  const bars=Object.entries(j.probabilities).sort((a,b)=>b[1]-a[1]).map(([n,v])=>
    `<div class="bar" style="--c:${(INFO[n]||k).c}"><div><span>${(INFO[n]||{t:n}).t}</span><b>${pct(v)}</b></div><div class="track"><div class="fill" data-w="${v*100}"></div></div></div>`).join("");
  $("out").innerHTML=`<div class="fade">
    <div class="verdict" style="--c:${k.c}"><div class="badge">${k.e}</div><div><h2>${k.t}</h2><p>${pct(j.confidence)} confidence · ${j.inference_time_ms} ms</p></div></div>
    ${bars}
    <div class="info"><b>What this means</b>${k.d}</div>
    ${j.confidence<0.7?'<div class="warn">⚠️ The model is not very sure. Try a closer, well-lit photo of a single leaf.</div>':""}
  </div>`;
  requestAnimationFrame(()=>requestAnimationFrame(()=>document.querySelectorAll(".fill").forEach(f=>f.style.width=f.dataset.w+"%")));
}
