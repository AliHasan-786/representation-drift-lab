import { useEffect, useMemo, useState } from "react";
import type {
  AggregateCheckpoint,
  BenchmarkArtifact,
  DetailedArtifact,
  DetailedCheckpoint,
  DatasetGalleryArtifact,
  DatasetRecord,
  DomainArtifact,
  DomainScenario,
  EarlyWarningArtifact,
  Interval,
  InterpolationArtifact,
  MethodArtifact,
  MethodRecord,
} from "./types";

const pct = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const fixed = (value: number, digits = 3) => value.toFixed(digits);
const staticAsset = (path: string) => {
  if (/^https?:\/\//.test(path)) return path;
  return `${import.meta.env.BASE_URL}${path.replace(/^\//, "")}`;
};
const usesAdapterPath = (method: MethodRecord) => [
  "standard-lora",
  "zscl-inspired-distillation",
  "retention-gradient-nullspace",
  "selective-v-lora",
].includes(method.id);
const fidelityLabel = (fidelity: string) => ({
  "exact-frozen-encoder-linear-classification-probe": "Frozen-encoder probe · exact",
  "joint-full-vision-encoder-and-random-head-baseline": "Full encoder + new head · baseline",
  "adapted-lp-ft-classification-baseline": "Linear-probe then fine-tune · adapted baseline",
  "compatible-wise-ft-source-baseline": "Source model for WiSE-FT comparison",
  "adapted-wise-ft-full-weight-space-ensemble": "Weight-space ensemble · WiSE-FT adapted",
  "corrected-local-pipeline-validation": "Corrected local pipeline validation",
  "inspired-baseline-not-exact-zscl": "ZSCL-inspired distillation · not a reproduction",
  "inspired-first-order-gradient-projection-baseline": "First-order gradient projection · inspired baseline",
  "selective-parameter-efficient-baseline": "Selective parameter-efficient baseline",
}[fidelity] ?? fidelity.replaceAll("-", " "));
const intervalText = (value: Interval) => `95% t-interval (unclipped): ${pct(value.ci_low)}–${pct(value.ci_high)} · n=${value.n}`;

function metric(value: Interval) {
  return (
    <>
      <strong>{pct(value.mean)}</strong>
      <span className="interval">{intervalText(value)}</span>
    </>
  );
}

function EvidenceChip({ children, manifestPath }: { children: React.ReactNode; manifestPath: string }) {
  return <a className="evidence-chip" href={staticAsset(manifestPath)} target="_blank" rel="noreferrer">{children}<span aria-hidden="true">↗</span></a>;
}

function Definition({ children }: { children: React.ReactNode }) {
  return <p className="definition">{children}</p>;
}

function moveTabFocus(
  event: React.KeyboardEvent<HTMLButtonElement>,
  index: number,
  count: number,
  select: (next: number) => void,
) {
  const offsets: Record<string, number> = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };
  let next: number | null = null;
  if (event.key in offsets) next = (index + offsets[event.key] + count) % count;
  if (event.key === "Home") next = 0;
  if (event.key === "End") next = count - 1;
  if (next === null) return;
  event.preventDefault();
  select(next);
  const tabs = event.currentTarget.parentElement?.querySelectorAll<HTMLElement>("[role='tab']");
  requestAnimationFrame(() => tabs?.[next!]?.focus());
}

function ClipExplainer({ gallery }: { gallery: DatasetGalleryArtifact }) {
  const examples = gallery.datasets.map((dataset) => ({ dataset, example: dataset.examples[0] }));
  const [selected, setSelected] = useState(0);
  const current = examples[selected];
  return (
    <div className="clip-explainer">
      <div className="clip-copy">
        <p className="eyebrow">The model, from zero</p>
        <h3>CLIP connects pictures with words</h3>
        <p><strong>CLIP</strong> stands for <strong>Contrastive Language–Image Pre-training</strong>. OpenAI trained it before this project on many image-and-text pairs. I started with that already-trained model; I did not build a vision model from nothing.</p>
        <p>CLIP has one path that turns an image into numbers and another that turns text into numbers. If the two number patterns point in similar directions, CLIP treats the picture and text as a good match.</p>
        <div className="clip-picker" aria-label="Choose a real dataset example">
          {examples.map(({ dataset }, index) => <button className={index === selected ? "active" : ""} onClick={() => setSelected(index)} key={dataset.id}>{dataset.name}</button>)}
        </div>
      </div>
      <div className="clip-machine" aria-live="polite">
        <figure><img src={staticAsset(current.example.src)} alt={current.example.alt} /><figcaption>Dataset label: <strong>{current.example.label}</strong></figcaption></figure>
        <div className="machine-arrow" aria-hidden="true">→</div>
        <div className="encoder-box"><span>Image encoder</span><strong>picture → numeric representation</strong><small>A representation is the model's internal description of what it sees.</small></div>
        <div className="machine-arrow" aria-hidden="true">↔</div>
        <div className="prompt-stack"><span>Text encoder compares prompts</span><strong>“a photo of {current.example.label}”</strong><small>The closest text representation becomes the predicted category.</small></div>
      </div>
      <div className="clip-facts">
        <div><span>ViT</span><p><strong>Vision Transformer:</strong> a neural network that reads an image as a sequence of small patches, much like a language model reads words.</p></div>
        <div><span>B/32</span><p><strong>Model size and patch size:</strong> “B” is the base-sized transformer; “32” means the image is divided into 32 × 32-pixel patches.</p></div>
        <div><span>Zero-shot</span><p><strong>Classification without new examples:</strong> CLIP can compare an image with written class names even before project-specific training.</p></div>
      </div>
    </div>
  );
}

function LoraExplainer() {
  const [adapted, setAdapted] = useState(true);
  return (
    <div className="lora-explainer">
      <div>
        <p className="eyebrow">How I changed CLIP</p>
        <h3>LoRA adds small, trainable side paths</h3>
        <p><strong>Fine-tuning</strong> means continuing to train an existing model for a more specific job. Changing every CLIP parameter would be expensive and could overwrite more of what it already knows.</p>
        <p><strong>LoRA</strong> means <strong>Low-Rank Adaptation</strong>. It freezes the original model and inserts small trainable matrices into selected attention layers. Those adapters learn the new task while the large original weights stay fixed.</p>
        <button className="state-toggle" aria-pressed={adapted} onClick={() => setAdapted(!adapted)}>{adapted ? "Show the untouched model" : "Show the adapted model"}</button>
      </div>
      <div className={`model-cutaway ${adapted ? "adapted" : "baseline"}`}>
        <div className="model-core"><span>Frozen CLIP core</span><strong>~87.6 million parameters</strong><small>kept unchanged</small></div>
        <div className="adapter-rail" aria-hidden={!adapted}>
          <i /><i /><i /><i /><i /><i />
          <span>LoRA adapters</span><strong>294,912 trainable parameters</strong>
        </div>
        <div className="parameter-meter"><span style={{ width: adapted ? "0.34%" : "0%" }} /><strong>{adapted ? "about 0.34% of the model trained" : "0% trained"}</strong></div>
        <p>{adapted ? "The food, satellite, or pet examples update only the highlighted adapter paths." : "This is the original zero-shot baseline used before any project-specific learning."}</p>
      </div>
    </div>
  );
}

function EvaluationFirewall() {
  return (
    <aside className="evaluation-firewall" aria-labelledby="firewall-title">
      <div className="firewall-copy">
        <p className="eyebrow">A fairness rule behind every score</p>
        <h3 id="firewall-title">The model cannot practice on its final exam</h3>
        <p>Imagine teaching from flashcards and then grading with the very same cards. A high score could mean real learning—or simply remembering the answers. This project keeps three image groups separate so the final score remains meaningful.</p>
      </div>
      <ol className="firewall-flow">
        <li><span>1</span><div><strong>New-skill practice</strong><p>Food, satellite, or pet images update the small LoRA adapters.</p></div></li>
        <li><span>2</span><div><strong>Optional memory reminder</strong><p>Two methods can use a separate everyday-object reference set to protect older knowledge.</p></div></li>
        <li><span>3</span><div><strong>Locked final check</strong><p>A different held-out object set measures retention. It is never used for an update or reminder.</p></div></li>
      </ol>
      <details>
        <summary>Why does this matter for this project?</summary>
        <p>An earlier version used the same retained images as both a training reference and an evaluation score for two methods. I treated that as a leakage risk, rebuilt those experiments with a disjoint reference split, and published the corrected runs. The method chart labels this as a resource trade-off rather than a free improvement.</p>
      </details>
    </aside>
  );
}

function DatasetCard({ dataset }: { dataset: DatasetRecord }) {
  return (
    <article className="dataset-card">
      <div className="dataset-card-heading"><div><span>{dataset.role}</span><h3>{dataset.name}</h3><p>{dataset.plain_name}</p></div><strong>{dataset.native_resolution}</strong></div>
      <div className="dataset-images">{dataset.examples.map((example) => <figure key={example.src}><img src={staticAsset(example.src)} alt={example.alt} loading="lazy" /><figcaption>{example.label}</figcaption></figure>)}</div>
      <p>{dataset.role_explanation}</p>
      <dl><div><dt>Full public dataset</dt><dd>{dataset.full_size}</dd></div><div><dt>Used in these local experiments</dt><dd>{dataset.local_protocol}</dd></div></dl>
      <a href={dataset.source_url} target="_blank" rel="noreferrer">Open dataset source ↗</a>
    </article>
  );
}

function DatasetLab({ gallery }: { gallery: DatasetGalleryArtifact }) {
  const scenarios = [
    { label: "Food ↔ everyday objects", ids: ["food101", "cifar10"] },
    { label: "Satellite ↔ 100 object types", ids: ["eurosat", "cifar100"] },
    { label: "Pet breeds ↔ handwritten digits", ids: ["pets", "mnist"] },
  ];
  const [selected, setSelected] = useState(0);
  const datasets = scenarios[selected].ids.map((id) => gallery.datasets.find((dataset) => dataset.id === id)!);
  return (
    <div className="dataset-lab">
      <div className="scenario-tabs" role="tablist" aria-label="Dataset scenarios">{scenarios.map((scenario, index) => <button id={`dataset-tab-${index}`} aria-controls="dataset-scenario-panel" tabIndex={index === selected ? 0 : -1} role="tab" aria-selected={index === selected} className={index === selected ? "active" : ""} onKeyDown={(event) => moveTabFocus(event, index, scenarios.length, setSelected)} onClick={() => setSelected(index)} key={scenario.label}>{scenario.label}</button>)}</div>
      <div className="dataset-pair" id="dataset-scenario-panel" role="tabpanel" aria-labelledby={`dataset-tab-${selected}`}><DatasetCard dataset={datasets[0]} /><div className="dataset-relationship"><span>train on the first</span><strong>then test both</strong><span>never train on the second</span></div><DatasetCard dataset={datasets[1]} /></div>
      <p className="gallery-note">These are real cached examples from the exact dataset repositories used by the experiment pipeline. Low-resolution CIFAR and MNIST images are enlarged with visible pixels so you can see what the model actually received.</p>
    </div>
  );
}

function WorkWalkthrough() {
  const steps = [
    ["Start with a baseline", "Run the untouched, already-trained CLIP model and record how well it recognizes both datasets before learning anything new."],
    ["Attach LoRA adapters", "Insert small trainable paths into the vision transformer's attention layers while freezing the rest of CLIP."],
    ["Teach the new task", "Show the model labeled examples from Food-101, EuroSAT, or Oxford Pets and update the LoRA parameters for 20 local training steps."],
    ["Save checkpoints", "Pause at steps 0, 5, 10, and 20. A checkpoint is a saved snapshot of the model at that moment."],
    ["Test learning and memory", "Measure new-task accuracy and retained-task accuracy at every checkpoint using examples that were not used for that update."],
    ["Look inside the model", "Compare internal representations against the baseline with CKA, layer maps, centroid movement, neighborhoods, and paired projections."],
    ["Repeat instead of trusting one run", "Use seeds 41, 42, and 43 so the result is not presented as if one random sample were universal."],
    ["Challenge the first story", "Run nine interventions and three domain pairs, preserve failures, and test whether drift really tracks forgetting."],
  ];
  const [selected, setSelected] = useState(0);
  return (
    <div className="work-walkthrough">
      <div className="work-step-list">{steps.map(([title], index) => <button className={index === selected ? "active" : ""} aria-pressed={index === selected} onClick={() => setSelected(index)} key={title}><span>{String(index + 1).padStart(2, "0")}</span>{title}</button>)}</div>
      <div className="work-step-detail"><span>Step {selected + 1} of {steps.length}</span><h3>{steps[selected][0]}</h3><p>{steps[selected][1]}</p><div className="step-progress"><i style={{ width: `${((selected + 1) / steps.length) * 100}%` }} /></div><button onClick={() => setSelected((selected + 1) % steps.length)}>{selected === steps.length - 1 ? "Start again" : "Next step"} →</button></div>
    </div>
  );
}

function CodeArchaeology() {
  const findings = [
    {
      label: "What appeared",
      title: "A historical code scaffold—not another result bundle",
      body: "The newly supplied DL Final folder contains CLIP/LoRA source code, tests, scripts, and two notebook templates. It contains no datasets, checkpoints, embeddings, result tables, logs, or executed notebook outputs. Source code can show what someone intended to build; only saved run evidence can show what actually happened.",
      evidence: "42 outer-folder files · 32 Python files · 0 stored notebook outputs",
      status: "lineage evidence",
    },
    {
      label: "Original intent",
      title: "The ambition was already broader than one chart",
      body: "The scaffold planned Food-101 adaptation, COCO and Flickr evaluation, seven dataset loaders, image/text drift, semantic clusters, drift speed, checkpoint animations, and an interactive Plotly dashboard. Its LoRA settings—rank 8, alpha 16, q_proj and v_proj—match a central design thread that survived into the rebuilt project.",
      evidence: "CLIP ViT-B/32 · rank-8 LoRA · cosine, centroid, Frechet, and alignment metrics",
      status: "implemented + planned code",
    },
    {
      label: "What did not run",
      title: "The advertised one-command pipeline could not complete as written",
      body: "The pipeline calls command options that its own scripts do not accept, requests an unregistered medical dataset, and invokes several files that explicitly say “not yet implemented.” The tests also import two dataset classes that do not exist. These are concrete source-level findings, not guesses based on missing output.",
      evidence: "7 explicit placeholders · incompatible downloader/training options · missing test imports",
      status: "verified blockers",
    },
    {
      label: "Scientific risks",
      title: "Several shortcuts could turn an attractive story into a wrong one",
      body: "Repeated Food-101 class captions were treated as if they were competing negatives. Internal drift alone was labeled catastrophic forgetting without requiring an accuracy loss. The early forecast could use later points from the same trajectory. And this branch evaluates COCO—not the CIFAR-10 path described in the submitted study—so it cannot be assumed to have produced that report.",
      evidence: "supervision semantics · performance/geometry conflation · future-information leakage · branch mismatch",
      status: "audit finding",
    },
    {
      label: "What I rebuilt",
      title: "The extension converts useful ideas into claims that can be checked",
      body: "The new system pins model and dataset revisions, records exact rows and seeds, rebuilds adapter topology before loading, uses same-class multi-positive supervision, separates performance from geometry, fits visual projections once, reports uncertainty, tests held-out warning logic, and rejects artifacts whose fingerprints change.",
      evidence: "43 Python checks · 3 independent seeds · 9 interventions · 3 published dataset pairs",
      status: "measured extension",
    },
  ];
  const [selected, setSelected] = useState(0);
  const finding = findings[selected];
  return (
    <div className="code-audit">
      <div className="audit-tabs" role="tablist" aria-label="Historical code audit">
        {findings.map((item, index) => <button id={`audit-tab-${index}`} aria-controls="audit-finding-panel" tabIndex={index === selected ? 0 : -1} role="tab" aria-selected={index === selected} className={index === selected ? "active" : ""} onKeyDown={(event) => moveTabFocus(event, index, findings.length, setSelected)} onClick={() => setSelected(index)} key={item.label}><span>{String(index + 1).padStart(2, "0")}</span>{item.label}</button>)}
      </div>
      <article className="audit-finding" id="audit-finding-panel" role="tabpanel" aria-labelledby={`audit-tab-${selected}`} aria-live="polite">
        <span className="audit-status">{finding.status}</span>
        <h3>{finding.title}</h3>
        <p>{finding.body}</p>
        <div><small>Evidence visible in the archive</small><strong>{finding.evidence}</strong></div>
      </article>
      <aside className="audit-rule"><span>Rule used throughout this project</span><strong>Code shows possibility.</strong><strong>Artifacts show execution.</strong><strong>Reproduction supports a claim.</strong></aside>
    </div>
  );
}

function ChartDecoder() {
  const outputs = [
    { id: "accuracy", label: "Accuracy lines", question: "Is the model getting better at the new task, and worse at the old one?", read: "Move left to right through training. Higher is better. The orange line is the new task; the teal line is the retained task.", caution: "A single percentage does not reveal how the internal model changed." },
    { id: "layers", label: "Layer map", question: "Where inside the 12-block vision transformer does change appear?", read: "Each cell is one transformer block. Brighter cells mean more change from the untouched baseline at that checkpoint.", caution: "More change is not automatically worse. A layer may change while accuracy stays stable." },
    { id: "embedding", label: "Embedding movement", question: "How did the model's internal map move for the same individual images?", read: "Each line connects one image before and after adaptation in the same fixed two-dimensional projection.", caution: "The screen shows a 2D summary of a much higher-dimensional representation." },
    { id: "confusion", label: "Confusion matrix", question: "Which categories does the adapted model mix up?", read: "Rows are true labels and columns are predictions. A strong diagonal means correct predictions; off-diagonal cells are mistakes.", caution: "The local class views contain few images and are diagnostic examples, not population estimates." },
  ];
  const [selected, setSelected] = useState(0);
  const output = outputs[selected];
  return (
    <div className="chart-decoder">
      <div className="decoder-tabs" role="tablist" aria-label="Visual output explanations">{outputs.map((item, index) => <button id={`decoder-tab-${index}`} aria-controls="decoder-panel" tabIndex={index === selected ? 0 : -1} role="tab" aria-selected={index === selected} onKeyDown={(event) => moveTabFocus(event, index, outputs.length, setSelected)} onClick={() => setSelected(index)} className={index === selected ? "active" : ""} key={item.id}>{item.label}</button>)}</div>
      <div className={`decoder-visual ${output.id}`} aria-hidden="true">
        {output.id === "accuracy" && <svg viewBox="0 0 420 220"><line x1="48" y1="15" x2="48" y2="190" /><line x1="48" y1="190" x2="400" y2="190" /><path className="demo-new" d="M50 150 L160 125 L270 85 L395 45" /><path className="demo-old" d="M50 70 L160 83 L270 78 L395 108" /><text x="8" y="28">higher</text><text x="315" y="215">training →</text></svg>}
        {output.id === "layers" && <div className="demo-layers">{Array.from({ length: 12 }, (_, index) => <i key={index} style={{ opacity: .12 + index * .07 }}><span>{index + 1}</span></i>)}</div>}
        {output.id === "embedding" && <svg viewBox="0 0 420 220">{[[60,50,95,70],[120,145,165,105],[225,65,270,45],[300,150,355,125],[180,180,225,160]].map((p,index)=><g key={index}><line x1={p[0]} y1={p[1]} x2={p[2]} y2={p[3]} /><circle cx={p[0]} cy={p[1]} r="6" /><circle className="after" cx={p[2]} cy={p[3]} r="7" /></g>)}</svg>}
        {output.id === "confusion" && <div className="demo-matrix">{Array.from({ length: 25 }, (_, index) => <i key={index} className={index % 6 === 0 ? "correct" : index % 7 === 0 ? "error" : ""} />)}</div>}
      </div>
      <div className="decoder-copy" id="decoder-panel" role="tabpanel" aria-labelledby={`decoder-tab-${selected}`}><p className="eyebrow">Question this visual answers</p><h3>{output.question}</h3><dl><div><dt>How to read it</dt><dd>{output.read}</dd></div><div><dt>What not to conclude</dt><dd>{output.caution}</dd></div></dl></div>
    </div>
  );
}

const GUIDE_ANSWERS = [
  { terms: ["what did", "actually do", "ali do", "project"], answer: "Ali began with an already-trained CLIP image-and-text model. He attached small LoRA adapters, trained those adapters on a new image domain, saved intermediate checkpoints, tested both the new task and an older retained task, measured internal representation change, repeated the experiment across seeds, methods, and dataset pairs, and rebuilt the work as a reproducible interactive system." },
  { terms: ["clip"], answer: "CLIP means Contrastive Language–Image Pre-training. It contains an image encoder and a text encoder trained to place matching pictures and descriptions near each other in a shared numeric space. This project starts from the existing CLIP ViT-B/32 model rather than training CLIP from scratch." },
  { terms: ["lora", "adapter", "fine-tun"], answer: "LoRA means Low-Rank Adaptation. Instead of changing all ~88 million CLIP parameters, the project freezes the original model and trains 294,912 small adapter parameters inserted into attention layers—about 0.34% of the model." },
  { terms: ["cifar", "food", "dataset", "mnist", "eurosat", "pets"], answer: "A dataset is a labeled collection of examples. Food-101, EuroSAT, and Oxford Pets are learning datasets in three separate scenarios. CIFAR-10, CIFAR-100, and MNIST are retained-task checks: the model is evaluated on them but does not use them for that scenario's training updates." },
  { terms: ["cka", "drift", "representation"], answer: "A representation is the model's internal numeric description of an image. CKA compares two collections of representations: 1 means highly similar. This project often shows 1 − CKA, so 0 means no measured change and larger values mean more drift. Drift is change, not automatically damage." },
  { terms: ["find", "result", "conclusion", "learn"], answer: "The most important conclusion is that internal drift and forgetting are not interchangeable. In one scenario the retained score fell, in another internal drift was much larger but the retained score barely fell, and in the Pets-to-MNIST scenario both the new and retained scores improved while representations still moved." },
  { terms: ["original", "report", "class", "extension"], answer: "The original 2025 class report introduced the CLIP + LoRA + Food-101/CIFAR-10 study and made stronger claims from a smaller experiment. The independent extension preserves that report as historical evidence, corrects and rebuilds the pipeline, adds uncertainty, methods, domains, failure cases, tests, provenance, and this interactive explanation." },
  { terms: ["new folder", "dl final", "source code", "code audit", "scaffold"], answer: "The newly supplied DL Final folder is historical source-code lineage, not a new result bundle. It contains CLIP/LoRA code, scripts, tests, and unexecuted notebook templates, but no datasets, checkpoints, embeddings, results, or logs. The audit found useful original ideas as well as concrete blockers and scientific risks; the extension preserves the ideas while rebuilding the evidence path." },
];

function localGuideAnswer(question: string) {
  const normalized = question.toLowerCase();
  const ranked = GUIDE_ANSWERS.map((item) => ({ ...item, score: item.terms.filter((term) => normalized.includes(term)).length })).sort((left, right) => right.score - left.score);
  return ranked[0].score ? ranked[0].answer : "Start with this mental model: the project teaches an already-trained image model a narrower new job, checks whether an older ability changes, and compares both visible performance and invisible internal representations. Try asking what CLIP, LoRA, a dataset, CKA, or the main result means.";
}

function localSensitiveInputReason(text: string) {
  const patterns: Array<[string, RegExp]> = [
    ["an email address", /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/i],
    ["a Social Security number", /\b\d{3}-\d{2}-\d{4}\b/],
    ["a payment-card number", /\b(?:\d[ -]?){13,19}\b/],
    ["a phone number", /(?:\+?\d[\s().-]*){10,15}/],
    ["an access credential", /\b(?:sk|rk|pk)[_-][A-Za-z0-9_-]{16,}\b/],
  ];
  return patterns.find(([, pattern]) => pattern.test(text))?.[0] ?? null;
}

function ProjectGuide() {
  const prompts = ["What did Ali actually do?", "Explain CLIP like I am new", "Why use two datasets?", "What did the new code folder add?", "What did the project find?"];
  const [question, setQuestion] = useState(prompts[0]);
  const [answer, setAnswer] = useState(localGuideAnswer(prompts[0]));
  const [mode, setMode] = useState("Guided answer · works offline");
  const [loading, setLoading] = useState(false);
  const [privacyNotice, setPrivacyNotice] = useState("");
  const ask = async (event?: React.FormEvent) => {
    event?.preventDefault();
    const cleanQuestion = question.trim().slice(0, 800);
    if (!cleanQuestion) return;
    const sensitiveReason = localSensitiveInputReason(cleanQuestion);
    if (sensitiveReason) {
      setAnswer(`For privacy, please remove ${sensitiveReason} and ask the project question again. This text was kept in your browser and was not sent to the GenAI guide.`);
      setMode("Guided answer · kept on this device");
      setPrivacyNotice(`Potential ${sensitiveReason} detected. It was not sent to the guide.`);
      return;
    }
    setPrivacyNotice("");
    setLoading(true);
    try {
      if (["localhost", "127.0.0.1"].includes(window.location.hostname)) throw new Error("local preview");
      const response = await fetch("/api/project-guide", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: cleanQuestion }) });
      if (!response.ok) throw new Error("guide unavailable");
      const payload = await response.json();
      setAnswer(payload.answer);
      setMode("GenAI answer · grounded in project evidence");
    } catch {
      setAnswer(localGuideAnswer(cleanQuestion));
      setMode("Guided answer · works offline");
    } finally {
      setLoading(false);
    }
  };
  return (
    <div className="project-guide">
      <div className="guide-intro"><p className="eyebrow">Interactive project guide</p><h3>Ask the question you think you “should already know.”</h3><p>No prerequisite is expected. The guide always has a curated, project-specific explanation available offline, so understanding the work never depends on an account, a model response, or a network connection.</p><div className="guide-prompts">{prompts.map((prompt) => <button onClick={() => { setQuestion(prompt); setAnswer(localGuideAnswer(prompt)); setMode("Guided answer · works offline"); }} key={prompt}>{prompt}</button>)}</div></div>
      <div className="guide-console"><form onSubmit={ask}><label htmlFor="project-question">Your question</label><textarea id="project-question" value={question} onChange={(event) => setQuestion(event.target.value)} rows={3} maxLength={800} aria-describedby="guide-privacy-note" /><button disabled={loading}>{loading ? "Thinking…" : "Ask the project"}</button></form>{privacyNotice && <p className="guide-privacy-notice" role="status">{privacyNotice}</p>}<div className="guide-answer" aria-live="polite"><span>{mode}</span><p>{answer}</p></div><small id="guide-privacy-note">Keep questions about this project. Do not include personal information or access credentials; apparent sensitive input stays in your browser and is not sent to the guide.</small></div>
    </div>
  );
}

function OriginalReportReader() {
  const [open, setOpen] = useState(false);
  return (
    <article className="original-report-card">
      <div><span className="report-status historical">Original submission · December 2025</span><h3>Detecting and Characterizing Representation Drift in Multimodal Vision-Language Models</h3><p>12-page Cornell Tech CS 5787 group report by Sahil Mhatre, Ali Hasan, and Corey Chen. This is the exact class submission, including its original figures, wording, implementation appendix, and conclusions.</p><div className="report-actions"><button className="button primary" onClick={() => setOpen(!open)}>{open ? "Close embedded reader" : "Read the report here"}</button><a className="button ghost" href={staticAsset("report/original-course-report.pdf")} target="_blank" rel="noreferrer">Open in a new tab</a><a className="button ghost" href={staticAsset("report/original-course-report.pdf")} download>Save PDF</a></div></div>
      {open && <iframe src={`${staticAsset("report/original-course-report.pdf")}#view=FitH&toolbar=1`} title="Original submitted class report" />}
    </article>
  );
}

function AccuracyChart({
  checkpoints,
  selected,
  onSelect,
  manifestPath,
}: {
  checkpoints: AggregateCheckpoint[];
  selected: number;
  onSelect: (index: number) => void;
  manifestPath: string;
}) {
  const width = 720;
  const height = 280;
  const padding = 42;
  const x = (index: number) =>
    padding + (index / Math.max(checkpoints.length - 1, 1)) * (width - padding * 2);
  const y = (value: number) => height - padding - ((value - 0.6) / 0.4) * (height - padding * 2);
  const path = (role: "retained" | "adaptation") =>
    checkpoints
      .map((checkpoint, index) =>
        `${index ? "L" : "M"}${x(index)},${y(checkpoint[role].top1_accuracy.mean)}`,
      )
      .join(" ");
  const summary = checkpoints
    .map(
      (checkpoint) =>
        `Step ${checkpoint.step}: retained ${pct(checkpoint.retained.top1_accuracy.mean)}, adapted ${pct(checkpoint.adaptation.top1_accuracy.mean)}`,
    )
    .join("; ");
  return (
    <figure className="chart-card wide">
      <div className="chart-heading">
        <div>
          <p className="eyebrow">Synchronized evidence</p>
          <h3>Learning and memory over training</h3>
        </div>
        <div className="legend" aria-label="Chart legend">
          <span><i className="line retained" /> Retained CIFAR-10</span>
          <span><i className="line adapted" /> Adapted Food-101</span>
        </div>
      </div>
      <EvidenceChip manifestPath={manifestPath}>Local tier · 20 updates · 3 seeds</EvidenceChip>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby="accuracy-title accuracy-desc">
        <title id="accuracy-title">Retained and adapted accuracy by checkpoint</title>
        <desc id="accuracy-desc">{summary}</desc>
        {[0.6, 0.7, 0.8, 0.9, 1].map((tick) => (
          <g key={tick}>
            <line className="grid" x1={padding} x2={width - padding} y1={y(tick)} y2={y(tick)} />
            <text className="axis-label" x={padding - 8} y={y(tick) + 4} textAnchor="end">{pct(tick, 0)}</text>
          </g>
        ))}
        <path className="series retained" d={path("retained")} />
        <path className="series adapted" d={path("adaptation")} />
        {checkpoints.map((checkpoint, index) => (
          <g key={checkpoint.step}>
            <line
              className={index === selected ? "cursor active" : "cursor"}
              x1={x(index)} x2={x(index)} y1={padding} y2={height - padding}
            />
            <circle
              className="point retained" cx={x(index)} cy={y(checkpoint.retained.top1_accuracy.mean)} r={index === selected ? 7 : 5}
            />
            <circle
              className="point adapted" cx={x(index)} cy={y(checkpoint.adaptation.top1_accuracy.mean)} r={index === selected ? 7 : 5}
            />
            <text className="axis-label" x={x(index)} y={height - 14} textAnchor="middle">{checkpoint.step}</text>
            <rect
              className="hit-target" x={x(index) - 25} y={padding} width="50" height={height - padding * 2}
              tabIndex={0} role="button" aria-label={`Select step ${checkpoint.step}`}
              onClick={() => onSelect(index)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") onSelect(index);
              }}
            />
          </g>
        ))}
      </svg>
      <Definition>“Learning” means improving on the new task; “memory” means preserving the old task. Researchers also call these plasticity and stability. Top-1 accuracy is the percentage of images whose first-choice label was correct.</Definition>
    </figure>
  );
}

function LayerMap({ checkpoint, manifestPath }: { checkpoint: AggregateCheckpoint; manifestPath: string }) {
  const layers = Object.entries(checkpoint.layerwise.retained)
    .filter(([, values]) => values.linear_cka)
    .sort(([left], [right]) => left.localeCompare(right));
  const maxLoss = Math.max(...layers.map(([, values]) => 1 - values.linear_cka.mean), 0.0001);
  return (
    <figure className="chart-card">
      <p className="eyebrow">Where change emerges</p>
      <h3>Vision encoder drift by layer</h3>
      <EvidenceChip manifestPath={manifestPath}>Local tier · selected checkpoint · 3 seeds</EvidenceChip>
      <div className="layer-map" role="list" aria-label={`Layer drift at step ${checkpoint.step}`}>
        {layers.map(([name, values], index) => {
          const loss = 1 - values.linear_cka.mean;
          const intensity = Math.max(0.08, loss / maxLoss);
          return (
            <div
              role="listitem" className="layer-cell" key={name}
              style={{ "--intensity": intensity } as React.CSSProperties}
              title={`${name}: CKA ${fixed(values.linear_cka.mean, 4)}`}
            >
              <span>{index + 1}</span>
              <strong>{fixed(loss, 4)}</strong>
              <small>{name.replace("vision_block_", "block ")}</small>
            </div>
          );
        })}
      </div>
      <Definition>Cells show 1 − linear CKA against the frozen baseline (higher means more representational change). Numeric labels prevent color-only interpretation.</Definition>
    </figure>
  );
}

function ParetoPlot({ benchmark, methods, interpolation }: { benchmark: BenchmarkArtifact; methods: MethodArtifact; interpolation: InterpolationArtifact }) {
  const first = benchmark.checkpoints[0];
  const points = [
    { label: "Zero-shot", x: first.retained.top1_accuracy.mean, y: first.adaptation.top1_accuracy.mean, kind: "baseline" },
    ...methods.methods.map((method) => ({
      label: method.label,
      x: method.metrics.final_retained_accuracy.mean,
      y: method.metrics.final_adaptation_accuracy.mean,
      kind: method.id,
    })),
    {
      label: "Scaled LoRA · α 0.5",
      x: interpolation.curve.find((point) => point.alpha === 0.5)!.retained.top1_accuracy.mean,
      y: interpolation.curve.find((point) => point.alpha === 0.5)!.adaptation.top1_accuracy.mean,
      kind: "posthoc-interpolation",
    },
  ];
  const colors = ["#126452", "#405b50", "#b65d42", "#2b8a83", "#874f91", "#1f7765", "#ff8a4c", "#5a8fd4", "#8f66b3", "#d2a72c", "#188b72"];
  const width = 520, height = 330, padding = 48;
  const x = (value: number) => padding + ((value - 0.58) / 0.22) * (width - padding * 2);
  const y = (value: number) => height - padding - ((value - 0.75) / 0.27) * (height - padding * 2);
  return (
    <figure className="chart-card">
      <p className="eyebrow">Trade-off space</p>
      <h3>Stability–plasticity frontier</h3>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby="pareto-title pareto-desc">
        <title id="pareto-title">Pareto comparison of retained and adapted accuracy</title>
        <desc id="pareto-desc">{points.map((point) => `${point.label} retains ${pct(point.x)} and adapts at ${pct(point.y)}`).join(". ")}.</desc>
        <line className="axis" x1={padding} x2={width - padding} y1={height - padding} y2={height - padding} />
        <line className="axis" x1={padding} x2={padding} y1={padding} y2={height - padding} />
        <text className="axis-title" x={width / 2} y={height - 10} textAnchor="middle">Retained accuracy →</text>
        <text className="axis-title" transform={`translate(14 ${height / 2}) rotate(-90)`} textAnchor="middle">Adapted accuracy →</text>
        {points.map((point, index) => (
          <g key={point.label}>
            <circle className={`pareto-point ${point.kind}`} style={{ fill: index ? colors[index] : "transparent", stroke: colors[index] }} cx={x(point.x)} cy={y(point.y)} r="9" />
            <text className="pareto-number" x={x(point.x)} y={y(point.y) + 3.5} textAnchor="middle">{index + 1}</text>
          </g>
        ))}
      </svg>
      <div className="pareto-key" aria-label="Pareto point values">
        {points.map((point, index) => <div key={point.label}><i style={{ background: index ? colors[index] : "transparent", borderColor: colors[index] }}>{index + 1}</i><span><strong>{point.label}</strong><small>retained {pct(point.x)} · adapted {pct(point.y)}</small></span></div>)}
      </div>
      <EvidenceChip manifestPath={methods.source_manifest.public_path}>Method-comparison tier · 200 updates · 3 seeds</EvidenceChip>
      <Definition>Both axes are mean accuracy across the same three seed-specific subsets; upper-right is better. 100% scores are tiny-subset ceiling effects, not perfect general performance. Method ranks remain preliminary because the probes are small.</Definition>
    </figure>
  );
}

function RecoveryCurve({ interpolation }: { interpolation: InterpolationArtifact }) {
  const [index, setIndex] = useState(2);
  const point = interpolation.curve[index];
  return (
    <div className="chart-card recovery-card">
      <div className="chart-heading">
        <div><p className="eyebrow">Post-hoc recovery · 3 seeds</p><h3>Scale the trained LoRA update</h3></div>
        <span className="status-chip">WiSE-FT-inspired</span>
      </div>
      <EvidenceChip manifestPath={interpolation.source_manifest.public_path}>Post-hoc local tier · 200 updates · 3 seeds</EvidenceChip>
      <label htmlFor="alpha-scale">Adapter output scale <strong>α = {point.alpha.toFixed(2)}</strong></label>
      <input id="alpha-scale" type="range" min="0" max={interpolation.curve.length - 1} step="1" value={index} onChange={(event) => setIndex(Number(event.target.value))} />
      <div className="ticks">{interpolation.curve.map((item) => <span key={item.alpha}>{item.alpha.toFixed(2)}</span>)}</div>
      <div className="recovery-metrics">
        <div><span>Retained</span><strong>{pct(point.retained.top1_accuracy.mean)}</strong><small>{intervalText(point.retained.top1_accuracy)}</small></div>
        <div><span>Adapted</span><strong>{pct(point.adaptation.top1_accuracy.mean)}</strong><small>{intervalText(point.adaptation.top1_accuracy)}</small></div>
        <div><span>CKA loss</span><strong>{fixed(1 - point.geometry.linear_cka.mean, 4)}</strong><small>baseline = 0</small></div>
      </div>
      <Definition>{interpolation.publication_caveat} Alpha was inspected on the same local evaluation scenario, so this is a recovery curve—not a held-out model-selection result.</Definition>
    </div>
  );
}

function EmbeddingMap({ checkpoint }: { checkpoint: DetailedCheckpoint }) {
  const points = checkpoint.samples.retained;
  const all = points.flatMap((point) => [point.baseline, point.current]);
  const xs = all.map((point) => point[0]);
  const ys = all.map((point) => point[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const project = ([px, py]: [number, number]) => [
    24 + ((px - minX) / Math.max(maxX - minX, 1e-9)) * 452,
    276 - ((py - minY) / Math.max(maxY - minY, 1e-9)) * 252,
  ];
  return (
    <figure className="chart-card">
      <p className="eyebrow">Fixed baseline PCA</p>
      <h3>Paired representation movement</h3>
      <svg viewBox="0 0 500 300" role="img" aria-label={`Paired CIFAR-10 representation movement at step ${checkpoint.step}`}>
        {points.map((point) => {
          const before = project(point.baseline);
          const after = project(point.current);
          return (
            <g key={point.id}>
              <line className="trajectory" x1={before[0]} y1={before[1]} x2={after[0]} y2={after[1]} />
              <circle className="embedding-before" cx={before[0]} cy={before[1]} r="3" />
              <circle className="embedding-after" cx={after[0]} cy={after[1]} r="4" />
            </g>
          );
        })}
      </svg>
      <Definition>Each line connects the same image before and after adaptation in PCA axes fit once on baseline embeddings. This is a paired trajectory, not independently fitted t-SNE.</Definition>
    </figure>
  );
}

function ClassMicroscope({ checkpoint }: { checkpoint: DetailedCheckpoint }) {
  const [selectedClass, setSelectedClass] = useState(0);
  const details = checkpoint.classwise.retained;
  const name = details.class_names[selectedClass];
  const row = details.confusion_matrix[selectedClass];
  const support = details.support[name];
  const movement = checkpoint.geometry.retained.class_centroid_movement as Record<string, number>;
  return (
    <div className="chart-card class-microscope">
      <div className="chart-heading">
        <div><p className="eyebrow">Error anatomy · single seed</p><h3>Class microscope</h3></div>
        <label>Class
          <select value={selectedClass} onChange={(event) => setSelectedClass(Number(event.target.value))}>
            {details.class_names.map((className, index) => <option value={index} key={className}>{className}</option>)}
          </select>
        </label>
      </div>
      <div className="microscope-stats">
        <div><span>Accuracy</span><strong>{pct(details.per_class_accuracy[name] ?? 0)}</strong></div>
        <div><span>Support</span><strong>{support}</strong></div>
        <div><span>Centroid movement</span><strong>{fixed(movement[String(selectedClass)], 3)}</strong></div>
      </div>
      <div className="confusion-list" aria-label={`Predictions for true class ${name}`}>
        {row.map((count, index) => (
          <div className="confusion-row" key={details.class_names[index]}>
            <span>{details.class_names[index]}</span>
            <i style={{ width: `${support ? (count / support) * 100 : 0}%` }} />
            <strong>{count}</strong>
          </div>
        ))}
      </div>
      <Definition>Confusion counts and class accuracy use the 30-image local seed-42 probe. They are diagnostic examples, not population-level class claims.</Definition>
    </div>
  );
}

function DomainStressTest({ artifact }: { artifact: DomainArtifact }) {
  const interpretations: Record<string, { title: string; body: string }> = {
    "food101-cifar10": {
      title: "The new skill improved, while the old score slipped.",
      body: "This is the classic forgetting pattern that motivated the project.",
    },
    "eurosat-cifar100": {
      title: "The model changed a lot inside, but lost little accuracy.",
      body: "Large representation drift did not translate into equally large forgetting.",
    },
    "pets-mnist": {
      title: "Both the new and retained tasks improved.",
      body: "Here drift accompanied positive transfer, showing that change is not automatically damage.",
    },
  };
  const bar = (scenario: DomainScenario, key: "adaptation_accuracy_change" | "retained_accuracy_change") => {
    const value = scenario.metrics[key].mean;
    const width = `${Math.min(Math.abs(value) / 0.35, 1) * 50}%`;
    return (
      <div className="domain-bar-track" aria-hidden="true">
        <i className={value >= 0 ? "positive" : "negative"} style={{ width, [value >= 0 ? "left" : "right"]: "50%" }} />
      </div>
    );
  };
  return (
    <>
      <EvidenceChip manifestPath={artifact.source_manifest.public_path}>Three local domain pairs · 20 updates · 3 seeds each</EvidenceChip>
      <div className="domain-grid">
        {artifact.scenarios.map((scenario) => {
        const interpretation = interpretations[scenario.id];
        return (
          <article className="domain-card" key={scenario.id}>
            <p className="eyebrow">{scenario.label}</p>
            <h3>{interpretation.title}</h3>
            <p>{interpretation.body}</p>
            <div className="domain-measure">
              <span>New-task accuracy change</span><strong>{scenario.metrics.adaptation_accuracy_change.mean >= 0 ? "+" : ""}{pct(scenario.metrics.adaptation_accuracy_change.mean)}</strong>
              {bar(scenario, "adaptation_accuracy_change")}
            </div>
            <div className="domain-measure">
              <span>Retained-task accuracy change</span><strong>{scenario.metrics.retained_accuracy_change.mean >= 0 ? "+" : ""}{pct(scenario.metrics.retained_accuracy_change.mean)}</strong>
              {bar(scenario, "retained_accuracy_change")}
            </div>
            <div className="domain-foot"><span>Internal change (1 − CKA)</span><strong>{fixed(scenario.metrics.retained_cka_loss.mean, 4)}</strong></div>
          </article>
        );
        })}
      </div>
    </>
  );
}

function ExpandedValidation({ artifact }: { artifact: BenchmarkArtifact }) {
  const baseline = artifact.checkpoints[0];
  const final = artifact.checkpoints.at(-1)!;
  const ckaLoss = 1 - final.geometry.retained.linear_cka.mean;
  return (
    <div className="expanded-validation">
      <div>
        <p className="eyebrow">Expanded confirmation check · 3 new seeds</p>
        <h3>The larger local probe reproduced the trade-off—not a universal rule</h3>
        <p>This preregistered check kept the same CLIP model and LoRA adapter, while increasing the Food-101 sample from 6 classes × 4 training images to 8 × 8, the retained CIFAR-10 check from 30 to 100 images, and the schedule from 20 to 50 updates.</p>
      </div>
      <EvidenceChip manifestPath={artifact.source_manifest.public_path}>Expanded local tier · 50 updates · 3 seeds</EvidenceChip>
      <div className="expanded-metrics">
        <article><span>New-task change</span><strong>+{pct(final.adaptation.accuracy_change.mean)}</strong><small>{intervalText(final.adaptation.accuracy_change)}</small></article>
        <article><span>Retained-task change</span><strong>{pct(final.retained.accuracy_change.mean)}</strong><small>{intervalText(final.retained.accuracy_change)}</small></article>
        <article><span>Internal change</span><strong>{fixed(ckaLoss, 4)}</strong><small>1 − CKA at step {final.step}</small></article>
      </div>
      <div className="expanded-read"><strong>How to read it:</strong><p>At step {final.step}, adaptation rose from {pct(baseline.adaptation.top1_accuracy.mean)} to {pct(final.adaptation.top1_accuracy.mean)} while retained accuracy fell from {pct(baseline.retained.top1_accuracy.mean)} to {pct(final.retained.top1_accuracy.mean)}. That supports the original local forgetting pattern under a larger fixed probe, but one backbone, one domain pair, and three seeds still do not establish a general law.</p></div>
      <Provenance benchmark={artifact} title="the expanded confirmation check" />
    </div>
  );
}

function Provenance({ benchmark, title }: { benchmark: BenchmarkArtifact; title: string }) {
  return (
    <details className="provenance">
      <summary>Inspect evidence behind {title}</summary>
      <dl>
        <div><dt>Evidence status</dt><dd>{benchmark.evidence_status}</dd></div>
        <div><dt>Run</dt><dd><code>{benchmark.run_id}</code></dd></div>
        <div><dt>Config</dt><dd><code>{benchmark.config_hash}</code></dd></div>
        <div><dt>Seeds</dt><dd>{benchmark.experiment.seeds.join(", ")}</dd></div>
        <div><dt>Model revision</dt><dd><code>{benchmark.experiment.model.resolved_revision}</code></dd></div>
        <div><dt>Manifest SHA-256</dt><dd><code>{benchmark.source_manifest.sha256}</code></dd></div>
      </dl>
    </details>
  );
}

function ReleaseGate({ benchmark }: { benchmark: BenchmarkArtifact }) {
  const [minimumRetention, setMinimumRetention] = useState(0.7);
  const [maximumDrift, setMaximumDrift] = useState(0.02);
  const final = benchmark.checkpoints.at(-1)!;
  const retainedAccuracy = final.retained.top1_accuracy.mean;
  const ckaLoss = 1 - final.geometry.retained.linear_cka.mean;
  const gates = [
    {
      label: "Memory floor",
      detail: `Retained accuracy must be at least ${pct(minimumRetention)}.`,
      observed: `${pct(retainedAccuracy)} measured`,
      passed: retainedAccuracy + 1e-9 >= minimumRetention,
    },
    {
      label: "Change limit",
      detail: `Internal CKA loss must be no more than ${fixed(maximumDrift, 3)}.`,
      observed: `${fixed(ckaLoss, 4)} measured`,
      passed: ckaLoss <= maximumDrift,
    },
    {
      label: "Repeatability floor",
      detail: "At least three independent seeds are required.",
      observed: `${benchmark.experiment.run_count} independent runs`,
      passed: benchmark.experiment.run_count >= 3,
    },
    {
      label: "Evidence maturity",
      detail: "Only confirmed, preregistered evidence may clear an update for use.",
      observed: benchmark.evidence_status.replaceAll("-", " "),
      passed: !benchmark.evidence_status.includes("preliminary"),
    },
  ];
  const numericGatesPass = gates.slice(0, 3).every((gate) => gate.passed);
  const approvalReady = gates.every((gate) => gate.passed);
  const reviewOutcome = approvalReady ? "Eligible for recorded approval" : "Human review required";
  const reviewBrief = [
    "REPRESENTATION DRIFT LAB — RELEASE REVIEW BRIEF",
    `Outcome: ${reviewOutcome}`,
    `Evidence status: ${benchmark.evidence_status}`,
    `Run: ${benchmark.run_id}`,
    `Configuration: ${benchmark.config_hash}`,
    `Checkpoint: step ${final.step}`,
    "",
    "Measured evidence",
    `- Retained accuracy: ${pct(retainedAccuracy)}`,
    `- Retained CKA loss: ${fixed(ckaLoss, 4)}`,
    `- Independent runs: ${benchmark.experiment.run_count}`,
    "",
    "Gate outcomes",
    ...gates.map((gate) => `- ${gate.label}: ${gate.passed ? "PASS" : "HOLD"} (${gate.observed}; ${gate.detail})`),
    "",
    approvalReady
      ? "Next action: a named owner must record the deployment scope, monitoring plan, and rollback conditions."
      : "Next action: retain human review; do not release automatically. Inspect scope, collect the missing evidence, and rerun the gate.",
  ].join("\n");
  return (
    <section className="release-gate" aria-labelledby="release-gate-title">
      <div className="release-gate-copy">
        <p className="eyebrow">Human-in-the-loop deployment guardrail</p>
        <h3 id="release-gate-title">A score is evidence—not permission to ship</h3>
        <p>Try two hypothetical thresholds. The calculator evaluates them against a real saved checkpoint, but it cannot approve a release. The evidence is still preliminary, so this update remains in human review even if the numeric checks pass.</p>
        <div className="release-controls">
          <label htmlFor="retention-floor">Minimum retained accuracy <strong>{pct(minimumRetention)}</strong></label>
          <input id="retention-floor" type="range" min="0.6" max="0.8" step="0.01" value={minimumRetention} onChange={(event) => setMinimumRetention(Number(event.target.value))} />
          <label htmlFor="drift-ceiling">Maximum CKA loss <strong>{fixed(maximumDrift, 3)}</strong></label>
          <input id="drift-ceiling" type="range" min="0.005" max="0.04" step="0.001" value={maximumDrift} onChange={(event) => setMaximumDrift(Number(event.target.value))} />
        </div>
      </div>
      <div className="release-gate-panel" aria-live="polite">
        <div className={`release-verdict ${approvalReady ? "approved" : "review"}`}>
          <span>{approvalReady ? "Eligible for recorded approval" : "Needs human review"}</span>
          <strong>{approvalReady ? "All release gates passed" : numericGatesPass ? "Evidence is not mature enough" : "At least one numeric gate failed"}</strong>
          <p>{approvalReady ? "A designated owner would still record a release decision and monitoring plan." : "No automatic release: inspect the run, decide whether the thresholds make sense for the intended use, and collect stronger evidence."}</p>
        </div>
        <ul className="release-checklist" aria-label="Release gate checks">
          {gates.map((gate) => <li key={gate.label} className={gate.passed ? "pass" : "hold"}><i aria-hidden="true">{gate.passed ? "✓" : "!"}</i><div><strong>{gate.label}</strong><span>{gate.detail}</span><small>{gate.observed}</small></div><b>{gate.passed ? "pass" : "hold"}</b></li>)}
        </ul>
      </div>
      <details className="release-brief">
        <summary>Open the deterministic review brief</summary>
        <p>This is the record a reviewer would inspect before recording a decision. It updates with the two illustrative thresholds above and deliberately omits a fake approval timestamp or signature.</p>
        <pre aria-label="Deterministic release review brief">{reviewBrief}</pre>
      </details>
      <p className="release-note"><strong>What this demonstrates:</strong> a deployment decision needs more than one accuracy number. It combines performance, internal change, repeatability, evidence quality, and accountable human review. The thresholds are illustrative—not a validated safety standard.</p>
    </section>
  );
}

function ImpactScopeAudit() {
  return (
    <section className="impact-scope-audit" aria-labelledby="impact-scope-title">
      <div>
        <p className="eyebrow">Scope and impact audit</p>
        <h3 id="impact-scope-title">This project tests models—not people</h3>
        <p>The experiments use public image datasets labeled mostly with objects, food, places, pets, and digits. They do not simulate a real decision about a person or service. That means a clean benchmark is useful evidence about model behavior, but it is not proof of fairness in a human setting.</p>
      </div>
      <div className="impact-scope-grid">
        <article><span>Checked here</span><strong>Evidence quality</strong><p>Separate image splits, repeat runs, transparent uncertainty, and a human release review help establish whether the measured benchmark result is trustworthy.</p></article>
        <article><span>Not assessed here</span><strong>Fairness across people</strong><p>There are no appropriate demographic labels, affected-user outcomes, or real decision context. Reporting a fairness score anyway would create false precision.</p></article>
        <article><span>Required in real use</span><strong>Human-impact review</strong><p>Define who is affected, test suitable outcome slices, keep meaningful decisions human-owned, watch feedback loops, and plan monitoring, appeals, and rollback.</p></article>
      </div>
      <details>
        <summary>Read the full fairness and feedback-loop scope audit</summary>
        <p>This interactive guide does not learn from visitor input. It is offline-first, and sensitive-looking questions are stopped in the browser before a provider request. Any future fairness claim needs a real decision context, appropriate consented evaluation data, an outcome measure, and a stated limit—not a borrowed benchmark metric.</p>
        <p>A real deployment review would also name the decision owner, identify who could be affected, preserve an appeal or correction path, prohibit automatic learning from complaints or overrides by default, and define monitoring and rollback before a trial begins.</p>
      </details>
    </section>
  );
}

function App() {
  const [benchmark, setBenchmark] = useState<BenchmarkArtifact | null>(null);
  const [earlyWarning, setEarlyWarning] = useState<EarlyWarningArtifact | null>(null);
  const [methods, setMethods] = useState<MethodArtifact | null>(null);
  const [interpolation, setInterpolation] = useState<InterpolationArtifact | null>(null);
  const [domains, setDomains] = useState<DomainArtifact | null>(null);
  const [expandedValidation, setExpandedValidation] = useState<BenchmarkArtifact | null>(null);
  const [datasetGallery, setDatasetGallery] = useState<DatasetGalleryArtifact | null>(null);
  const [detail, setDetail] = useState<DetailedArtifact | null>(null);
  const [selected, setSelected] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [expandedLoading, setExpandedLoading] = useState(false);
  const [expandedError, setExpandedError] = useState<string | null>(null);
  const [driftThreshold, setDriftThreshold] = useState(0.008);

  const loadCore = () => {
    setError(null);
    Promise.all([
      fetch(staticAsset("data/benchmark-local.json")).then((response) => {
        if (!response.ok) throw new Error("Benchmark artifact unavailable");
        return response.json();
      }),
      fetch(staticAsset("data/early-warning-methodology.json")).then((response) => {
        if (!response.ok) throw new Error("Early-warning artifact unavailable");
        return response.json();
      }),
      fetch(staticAsset("data/method-comparison-local.json")).then((response) => {
        if (!response.ok) throw new Error("Method artifact unavailable");
        return response.json();
      }),
      fetch(staticAsset("data/interpolation-local.json")).then((response) => {
        if (!response.ok) throw new Error("Interpolation artifact unavailable");
        return response.json();
      }),
      fetch(staticAsset("data/domain-comparison-local.json")).then((response) => {
        if (!response.ok) throw new Error("Domain artifact unavailable");
        return response.json();
      }),
      fetch(staticAsset("data/dataset-gallery.json")).then((response) => {
        if (!response.ok) throw new Error("Dataset gallery unavailable");
        return response.json();
      }),
    ])
      .then(([benchmarkPayload, warningPayload, methodPayload, interpolationPayload, domainPayload, galleryPayload]) => {
        setBenchmark(benchmarkPayload);
        setEarlyWarning(warningPayload);
        setMethods(methodPayload);
        setInterpolation(interpolationPayload);
        setDomains(domainPayload);
        setDatasetGallery(galleryPayload);
      })
      .catch((reason: Error) => setError(reason.message));
  };

  useEffect(loadCore, []);

  const loadDetail = () => {
    setDetailLoading(true);
    fetch(staticAsset("data/reproduction-local.json"))
      .then((response) => {
        if (!response.ok) throw new Error("Detailed artifact unavailable");
        return response.json();
      })
      .then(setDetail)
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setDetailLoading(false));
  };

  const loadExpandedValidation = () => {
    setExpandedLoading(true);
    setExpandedError(null);
    fetch(staticAsset("data/benchmark-expanded-local.json"))
      .then((response) => {
        if (!response.ok) throw new Error("Expanded validation artifact unavailable");
        return response.json();
      })
      .then(setExpandedValidation)
      .catch((reason: Error) => setExpandedError(reason.message))
      .finally(() => setExpandedLoading(false));
  };

  const stoppingCheckpoint = useMemo(() => {
    if (!benchmark) return null;
    return [...benchmark.checkpoints]
      .reverse()
      .find((checkpoint) => 1 - checkpoint.geometry.retained.linear_cka.mean <= driftThreshold)
      ?? benchmark.checkpoints[0];
  }, [benchmark, driftThreshold]);

  if (error) {
    return <main id="main" className="load-state"><p className="eyebrow">Artifact error</p><h1>The evidence bundle could not load.</h1><p>{error}</p><button onClick={loadCore}>Retry</button></main>;
  }
  if (!benchmark || !earlyWarning || !methods || !interpolation || !domains || !datasetGallery) {
    return <main id="main" className="load-state" aria-busy="true"><div className="loader" /><p>Validating experiment evidence…</p></main>;
  }

  const checkpoint = benchmark.checkpoints[selected];
  const detailedCheckpoint = detail?.checkpoints.find((item) => item.step === checkpoint.step);
  const finalCheckpoint = benchmark.checkpoints.at(-1)!;
  const warningMetrics = earlyWarning.evaluation.test_metrics;

  return (
    <>
      <header className="site-header">
        <a className="wordmark" href="#top"><span>RD</span> Representation Drift Lab</a>
        <nav aria-label="Project sections">
          <a href="#foundations">Start at zero</a><a href="#datasets">See the data</a><a href="#explore">Experiment</a><a href="#twist">The twist</a><a href="#guide">Ask</a><a href="#reports">Reports</a>
        </nav>
        <a className="report-link" href={staticAsset("report/original-course-report.pdf")} target="_blank" rel="noreferrer">Class report · PDF</a>
      </header>

      <main id="main">
        <section className="hero" id="top">
          <div className="hero-copy">
            <p className="eyebrow">Ali Hasan · independent post-course extension</p>
            <h1>Can teaching an AI something new make it <em>forget?</em></h1>
            <p className="hero-lede">
              I taught an existing image-and-text model to recognize new kinds of images, then checked whether it became worse at things it already knew. This page explains every part of that process from the beginning—no AI background assumed.
            </p>
            <div className="hero-actions"><a className="button primary" href="#foundations">Start from absolute zero</a><a className="button ghost" href="#datasets">Show me the images</a></div>
            <p className="evidence-banner"><span /> Preliminary local evidence · 3 independent seeds · every number traceable</p>
          </div>
          <div className="hero-result" aria-label="Primary local result">
            <p>One measured example after 20 small training updates</p>
            <div className="result-number">+{pct(finalCheckpoint.adaptation.accuracy_change.mean)}</div>
            <span>better at the new food-recognition task</span>
            <hr />
            <div className="result-number negative">{pct(finalCheckpoint.retained.accuracy_change.mean)}</div>
            <span>worse at the retained everyday-object task</span>
            <small>Later sections explain Food-101, CIFAR-10, LoRA, “training steps,” and exactly how these percentages were produced.</small>
            <EvidenceChip manifestPath={benchmark.source_manifest.public_path}>Local tier · 20 updates · 3 seeds</EvidenceChip>
          </div>
        </section>

        <section className="scope-strip" aria-label="Project scope">
          <div><span>Plain-English question</span><strong>Can a model learn a new skill without damaging an old one?</strong></div>
          <div><span>What I changed</span><strong>Small adapters inside an existing image model</strong></div>
          <div><span>What the evidence means</span><strong>A careful local case study—not a universal law about AI</strong></div>
        </section>

        <section className="section foundations" id="foundations">
          <div className="section-heading">
            <div><p className="eyebrow">01 · Assume nothing</p><h2>Start with the basic objects in this story</h2></div>
            <p>An AI model is software that learns patterns from examples. An image-classification model receives a picture and chooses a category such as “cat,” “ship,” or “hot and sour soup.” Training changes the model using labeled examples; evaluation tests it on examples it did not update from.</p>
          </div>
          <div className="concept-grid">
            <article><span>1</span><h3>Model</h3><p>A large collection of adjustable numbers—parameters—that turns an input into an output. Here the input is an image and the output is a category prediction.</p></article>
            <article><span>2</span><h3>Dataset</h3><p>An organized collection of examples and labels. “Image of a frog” plus the label “frog” is one labeled example.</p></article>
            <article><span>3</span><h3>Training</h3><p>Show examples, measure mistakes, and slightly adjust parameters so future predictions improve. One adjustment cycle is a training step.</p></article>
            <article><span>4</span><h3>Evaluation</h3><p>Test the model on held-out examples without changing it. This separates “did it learn?” from “did we merely show it the answers?”</p></article>
          </div>
          <ClipExplainer gallery={datasetGallery} />
          <LoraExplainer />
          <EvaluationFirewall />
          <div className="section-heading compact-heading"><div><p className="eyebrow">What I actually did</p><h2>The project as eight concrete actions</h2></div><p>This is the complete workflow stripped of research vocabulary. Select each step to see where training, testing, measurement, and the later extension fit.</p></div>
          <WorkWalkthrough />
          <div className="plain-callout"><strong>The key distinction:</strong><p><strong>drift</strong> means the model's internal numeric descriptions changed. <strong>Forgetting</strong> means its measured performance became worse. They can happen together, separately, or while another capability improves.</p></div>
        </section>

        <section className="section datasets-section" id="datasets">
          <div className="section-heading"><div><p className="eyebrow">02 · See the evidence inputs</p><h2>What are Food-101, CIFAR, EuroSAT, Pets, and MNIST?</h2></div><p>They are public labeled image datasets—not models and not methods. Each experiment pairs one <strong>learning dataset</strong> with one separate <strong>memory check</strong>. Change the scenario below to inspect real examples and their roles.</p></div>
          <DatasetLab gallery={datasetGallery} />
        </section>

        <section className="section visual-school" id="visual-school">
          <div className="section-heading"><div><p className="eyebrow">03 · Learn the visual language</p><h2>What does each output show—and how should I read it?</h2></div><p>The diagrams below teach the grammar first. After that, the experiment explorer uses the same visual forms with measured data. Nothing should require guessing what an axis, color, cell, or line means.</p></div>
          <ChartDecoder />
        </section>

        <section className="section guide-section" id="guide">
          <div className="section-heading"><div><p className="eyebrow">Ask instead of pretending</p><h2>Use the project guide whenever something is unclear</h2></div><p>The guide is intentionally positioned before the measured outputs. It answers in plain language first and can be revisited after the technical sections make more sense.</p></div>
          <ProjectGuide />
        </section>

        <section className="section" id="explore">
          <div className="section-heading">
            <div><p className="eyebrow">04 · Now use the real outputs</p><h2>Watch the model learn—and check what it keeps</h2></div>
            <p>A <strong>checkpoint</strong> is a saved moment during training. Move the control to synchronize the scores and internal measurements. “Accuracy” means the percentage of held-out images assigned the correct label.</p>
          </div>
          <div className="metric-primer" aria-label="How to read the experiment numbers">
            <article><strong>Δ means “change”</strong><p>A positive value went up from the starting model; a negative value went down.</p></article>
            <article><strong>n = 3 means three runs</strong><p>The experiment was repeated with three different controlled random starting conditions, called seeds.</p></article>
            <article><strong>Unclipped t-interval shows uncertainty</strong><p>Each 95% interval summarizes variation across three runs. It is intentionally not clipped to 0–100%, so a bound can exceed those limits when the sample is small; the mean is still an accuracy percentage.</p></article>
            <article><strong>CKA compares internals</strong><p>Centered Kernel Alignment compares two sets of internal representations. Here, 1.0 means identical and a lower value means more change.</p></article>
          </div>
          <div className="timeline-control">
            <label htmlFor="checkpoint">Training checkpoint <strong>step {checkpoint.step}</strong></label>
            <input id="checkpoint" type="range" min="0" max={benchmark.checkpoints.length - 1} step="1" value={selected} onChange={(event) => setSelected(Number(event.target.value))} />
            <div className="ticks">{benchmark.checkpoints.map((item) => <span key={item.step}>{item.step}</span>)}</div>
          </div>
          <div className="metric-grid">
            <article><span>Old-task accuracy (memory)</span>{metric(checkpoint.retained.top1_accuracy)}</article>
            <article><span>New-task accuracy (learning)</span>{metric(checkpoint.adaptation.top1_accuracy)}</article>
            <article><span>Internal similarity (CKA)</span><strong>{fixed(checkpoint.geometry.retained.linear_cka.mean, 4)}</strong><span className="interval">1.0 means unchanged · n=3</span></article>
            <article><span>Image–word match change (Δ)</span><strong>{fixed(checkpoint.cross_modal.retained.alignment_change.mean, 4)}</strong><span className="interval">average paired similarity change · n=3</span></article>
          </div>
          <div className="visual-grid"><AccuracyChart checkpoints={benchmark.checkpoints} selected={selected} onSelect={setSelected} manifestPath={benchmark.source_manifest.public_path} /><LayerMap checkpoint={checkpoint} manifestPath={benchmark.source_manifest.public_path} /></div>
          <div className="insight-callout"><span>How to read step {checkpoint.step}</span><p>The new Food-101 task changed by <strong>{checkpoint.adaptation.accuracy_change.mean >= 0 ? "+" : ""}{pct(checkpoint.adaptation.accuracy_change.mean)}</strong>; retained CIFAR-10 accuracy changed by <strong>{checkpoint.retained.accuracy_change.mean >= 0 ? "+" : ""}{pct(checkpoint.retained.accuracy_change.mean)}</strong>; and internal CKA similarity is <strong>{fixed(checkpoint.geometry.retained.linear_cka.mean, 4)}</strong>. Read the accuracy and geometry together: one does not replace the other.</p></div>
          <Provenance benchmark={benchmark} title="the checkpoint explorer" />

          {!detail && <div className="lazy-panel"><div><p className="eyebrow">Detailed evidence · 848 KB on demand</p><h3>Open paired embeddings and class errors</h3><p>The initial page stays compact. Load the single-seed detail only when you want to inspect individual classes and fixed-projection sample movement.</p></div><button onClick={loadDetail} disabled={detailLoading}>{detailLoading ? "Loading…" : "Load microscope"}</button></div>}
          {detailedCheckpoint && <div className="visual-grid detail-grid"><EmbeddingMap checkpoint={detailedCheckpoint} /><ClassMicroscope checkpoint={detailedCheckpoint} /></div>}
        </section>

        <section className="section domain-section twist-section" id="twist">
          <div className="section-heading"><div><p className="eyebrow">05 · The twist</p><h2>The first result was real. The simple explanation was not.</h2></div><p>The Food-101 → CIFAR-10 experiment showed one familiar trade-off: the model learned a new skill while its old score slipped. The next question was whether internal change reliably explained that pattern. I tested that assumption before treating it as a conclusion.</p></div>
          <div className="twist-statement"><div><span>What one pair suggested</span><strong>“More internal change may mean more forgetting.”</strong></div><i aria-hidden="true">→</i><div><span>What three pairs showed</span><strong>Change could accompany forgetting, little performance loss, or improvement.</strong></div></div>
          <DomainStressTest artifact={domains} />
          <div className="plain-callout"><strong>What changed after adding more domains:</strong><p>the original “drift predicts forgetting” story became less convincing. Across these scenarios, similar concepts behaved differently, so drift is best treated as diagnostic evidence—not a verdict.</p></div>
          {!expandedValidation && <div className="lazy-panel expanded-loader"><div><p className="eyebrow">Expanded fixed-split validation · 204 KB on demand</p><h3>Check the larger Food-101 → CIFAR-10 probe</h3><p>The original local run is intentionally small. This preregistered three-seed confirmation check increases both the adaptation and retained evaluation samples without changing the model family or adapter method.</p></div><button onClick={loadExpandedValidation} disabled={expandedLoading}>{expandedLoading ? "Loading…" : "Load expanded confirmation"}</button></div>}
          {expandedError && <p className="expanded-error" role="status">The expanded artifact could not load: {expandedError}</p>}
          {expandedValidation && <ExpandedValidation artifact={expandedValidation} />}
          <details className="provenance"><summary>Inspect evidence behind the domain stress test</summary><dl><div><dt>Evidence status</dt><dd>{domains.evidence_status}</dd></div><div><dt>Run</dt><dd><code>{domains.run_id}</code></dd></div><div><dt>Config</dt><dd><code>{domains.config_hash}</code></dd></div><div><dt>Scenarios</dt><dd>{domains.scenarios.length} × 3 seeds</dd></div><div><dt>Publication caveat</dt><dd>{domains.publication_caveat}</dd></div><div><dt>Manifest SHA-256</dt><dd><code>{domains.source_manifest.sha256}</code></dd></div></dl></details>
        </section>

        <section className="section tinted" id="methods">
          <div className="section-heading"><div><p className="eyebrow">06 · Compare interventions</p><h2>There is no single “best” way to adapt</h2></div><p>After the twist, the trade-off space becomes more useful: an intervention is a different choice for what to train or how to protect the older capability. The upper-right of the chart means better performance on both tasks. The detailed names are introduced only after the basic finding and its limit are established.</p></div>
          <div className="decision-grid">
            <ParetoPlot benchmark={benchmark} methods={methods} interpolation={interpolation} />
            <div className="chart-card simulator">
              <p className="eyebrow">Recorded-checkpoint simulator</p><h3>Drift-aware early stop</h3>
              <label htmlFor="drift-limit">Maximum 1 − CKA <strong>{fixed(driftThreshold, 3)}</strong></label>
              <input id="drift-limit" type="range" min="0" max="0.015" step="0.001" value={driftThreshold} onChange={(event) => setDriftThreshold(Number(event.target.value))} />
              {stoppingCheckpoint && <div className="stop-result"><span>Stop at</span><strong>step {stoppingCheckpoint.step}</strong><dl><div><dt>Retained</dt><dd>{pct(stoppingCheckpoint.retained.top1_accuracy.mean)}</dd></div><div><dt>Adapted</dt><dd>{pct(stoppingCheckpoint.adaptation.top1_accuracy.mean)}</dd></div><div><dt>CKA loss</dt><dd>{fixed(1 - stoppingCheckpoint.geometry.retained.linear_cka.mean, 4)}</dd></div></dl></div>}
              <Definition>This policy is retrospective and constrained to four observed checkpoints. It does not validate CKA as a causal control signal.</Definition>
            </div>
          </div>
          <RecoveryCurve interpolation={interpolation} />

          <div className="matrix-wrap">
            <div className="chart-heading"><div><p className="eyebrow">Method matrix</p><h3>What has actually been run</h3></div><span className="status-chip">Benchmark expansion active</span></div>
            <table><thead><tr><th>Method</th><th>Adapted accuracy</th><th>Retained accuracy</th><th>CKA loss</th><th>Trainable params</th><th>Train cost</th><th>Inference cost</th><th>Status</th></tr></thead>
              <tbody>
                <tr><th>Frozen zero-shot</th><td>{pct(benchmark.checkpoints[0].adaptation.top1_accuracy.mean)}</td><td>{pct(benchmark.checkpoints[0].retained.top1_accuracy.mean)}</td><td>0</td><td>0</td><td>evaluation only</td><td>1× encoder</td><td><span className="tag complete">measured</span></td></tr>
                {methods.methods.map((method) => <tr key={method.id}><th>{method.label}<small className="method-fidelity" title={method.fidelity}>{fidelityLabel(method.fidelity)}</small>{method.retained_reference.used && <small className="method-resource">uses a separate memory-reference set</small>}</th><td>{pct(method.metrics.final_adaptation_accuracy.mean)}<small className="table-interval">{intervalText(method.metrics.final_adaptation_accuracy)}</small></td><td>{pct(method.metrics.final_retained_accuracy.mean)}<small className="table-interval">{intervalText(method.metrics.final_retained_accuracy)}</small></td><td>{fixed(method.metrics.retained_cka_loss.mean, 4)}</td><td>{Math.round(method.metrics.trainable_parameters.mean).toLocaleString()}</td><td>{method.training_budget.probe_steps ? `${method.training_budget.probe_steps} probe + ` : ""}{method.training_budget.joint_or_adaptation_steps} joint/adapt</td><td>{usesAdapterPath(method) ? "adapter path" : "task head"}</td><td><span className="tag complete">3 seeds</span></td></tr>)}
                <tr><th>Scaled LoRA · α 0.5<small className="method-fidelity">WiSE-FT-inspired adapter-output interpolation</small></th><td>{pct(interpolation.curve[2].adaptation.top1_accuracy.mean)}</td><td>{pct(interpolation.curve[2].retained.top1_accuracy.mean)}</td><td>{fixed(1 - interpolation.curve[2].geometry.linear_cka.mean, 4)}</td><td>0 additional</td><td>post-hoc</td><td>scaled adapter</td><td><span className="tag exploratory">exploratory</span></td></tr>
              </tbody>
            </table>
          </div>
        </section>

        <section className="section" id="research">
          <div className="section-heading"><div><p className="eyebrow">07 · Test the hypothesis honestly</p><h2>Can early drift forecast final forgetting?</h2></div><p>A useful warning system must work on scenarios it never saw during training. I built that held-out evaluation protocol—including simple comparison rules and checks for whether predictions match reality—but the current predictor data is synthetic. It proves the evaluator works, not that real forgetting is predictable.</p></div>
          <div className="warning-banner"><strong>{earlyWarning.evidence_status.replaceAll("-", " ")}</strong><span>{earlyWarning.publication_caveat}</span></div>
          <div className="metric-grid warning-metrics">
            <article><span>Held-out model RMSE</span><strong>{fixed(warningMetrics.early_warning_model.rmse, 3)}</strong><span className="interval">final forgetting units</span></article>
            <article><span>Train-mean baseline RMSE</span><strong>{fixed(warningMetrics.train_mean_baseline.rmse, 3)}</strong><span className="interval">naive baseline</span></article>
            <article><span>Persistence baseline RMSE</span><strong>{fixed(warningMetrics.early_forgetting_persistence_baseline.rmse, 3)}</strong><span className="interval">early = final assumption</span></article>
            <article><span>Test calibration slope</span><strong>{fixed(warningMetrics.early_warning_model.calibration_slope ?? 0, 2)}</strong><span className="interval">ideal = 1.0</span></article>
          </div>
          <div className="failure-grid">
            <article><span className="case-number">01</span><h3>Geometry moved; accuracy did not</h3><p>In seed 42, retained accuracy returned to its {pct(0.7333)} baseline at step 20 while retained CKA fell to 0.9882. Drift and forgetting are not interchangeable.</p><small>Observed local case · config a3410b237d0c5b79</small></article>
            <article><span className="case-number">02</span><h3>Performance briefly improved during drift</h3><p>At step 10, the same run's retained accuracy rose by 3.3 points even as its representation departed from baseline. A scalar drift alarm can miss sign and task relevance.</p><small>Observed local case · seed 42</small></article>
            <article><span className="case-number">03</span><h3>The legacy forecast failed</h3><p>The original step-8,000 forecast predicted final cosine drift of 0.6647. The saved artifact ended at 0.3055—an absolute error of 0.3592.</p><small>Historical negative result · source bundle registered</small></article>
            <article><span className="case-number">04</span><h3>A fourth pair failed its validity check</h3><p>An exploratory Pets-to-EuroSAT run began at 0% retained accuracy in all three seeds. Because a floor cannot reveal further forgetting, I preserved the artifact but excluded it from the three-scenario comparison.</p><small>Excluded run · zero-score floor · no scientific claim</small></article>
          </div>
        </section>

        <section className="section dark" id="reproduce">
          <div className="section-heading"><div><p className="eyebrow">08 · Engineering and reproducibility</p><h2>Evidence that can be rerun</h2></div><p>This is a research system, not only a notebook. Each setup gets an identity; interrupted runs can resume safely; dataset selections are recorded; outputs can regenerate metrics without retraining; and public files are checked against fingerprints that expose accidental changes.</p></div>
          <ReleaseGate benchmark={benchmark} />
          <ImpactScopeAudit />
          <div className="engineering-grid">
            <article><strong>43</strong><span>automated Python checks</span><p>Metrics, disjoint splits, caching, loss semantics, resume safety, provenance, and artifact aggregation.</p></article>
            <article><strong>294,912</strong><span>trainable parameters</span><p>All non-LoRA parameters are frozen and the invariant is checked at runtime.</p></article>
            <article><strong>Byte-stable</strong><span>metric regeneration</span><p>Saved embeddings reproduce the same public derivative without retraining.</p></article>
          </div>
          <pre aria-label="Reproduction commands"><code>{`make test\nmake reproduce-local\nPYTHONPATH=src .venv/bin/python -m driftlab benchmark-clip \\\n  --suite configs/reproduction-local-multiseed.yaml --regenerate-metrics`}</code></pre>
          <Provenance benchmark={benchmark} title="the published benchmark" />
        </section>

        <section className="section audit-section" id="code-audit">
          <div className="section-heading"><div><p className="eyebrow">09 · Code archaeology</p><h2>What the newly recovered source folder changes</h2></div><p>I audited every file in the supplied <code>DL Final</code> folder. It adds valuable history, not new performance numbers. Use the tabs to follow the boundary between an idea, code, an executed artifact, and a reproducible claim.</p></div>
          <CodeArchaeology />
          <div className="plain-callout"><strong>Why include imperfections?</strong><p>Because understanding and improving earlier work is part of the project. Hiding incomplete branches would make the story simpler but less accurate. The audit shows which ideas were preserved, which claims were rejected, and exactly why the independent extension exists.</p></div>
        </section>

        <section className="section narrative">
          <div><p className="eyebrow">Methodology</p><h2>What changed from the course project</h2><p>The 2025 group project established the question using CLIP, Food-101, CIFAR-10, and LoRA. This independent extension rebuilds that work around traceability, correct multi-positive supervision, fixed projections, stable covariance estimates, layerwise diagnostics, cross-modal separation, independent seeds, and deployment-safe evidence artifacts.</p></div>
          <div><p className="eyebrow">Limitations</p><h2>What this does not prove</h2><p>The local runs are deliberately small and still use one CLIP backbone. Confidence intervals quantify seed variation in these bounded settings only. Association is not causation; synthetic predictor validation is not evidence of real-world predictability; class-level observations are underpowered.</p></div>
          <div><p className="eyebrow">Attribution</p><h2>Coursework and independent work</h2><p>The original Cornell Tech CS 5787 project was completed by Sahil Mhatre, Ali Hasan, and Corey Chen. The repository architecture, corrected pipeline, diagnostics, multi-seed protocol, interactive experience, and post-course experiments shown here are Ali Hasan's independent extension.</p></div>
          <div><p className="eyebrow">Primary sources</p><h2>Research lineage</h2><ul className="bibliography"><li><a href="https://arxiv.org/abs/2103.00020">CLIP</a> · Radford et al., 2021</li><li><a href="https://arxiv.org/abs/2106.09685">LoRA</a> · Hu et al., 2021</li><li><a href="https://proceedings.mlr.press/v97/kornblith19a.html">CKA</a> · Kornblith et al., 2019</li><li><a href="https://arxiv.org/abs/2202.10054">Feature distortion and LP-FT</a> · Kumar et al., 2022</li><li><a href="https://openaccess.thecvf.com/content/CVPR2022/html/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.html">WiSE-FT</a> · Wortsman et al., 2022</li></ul></div>
        </section>

        <section className="section reports-section" id="reports">
          <div className="section-heading"><div><p className="eyebrow">10 · The written record</p><h2>The submitted class report and the independent extension</h2></div><p>These are deliberately separate. The original PDF is preserved exactly as submitted by the three-person course team. The extension report documents the later audit, rebuilt system, added experiments, contradictions, and limitations.</p></div>
          <div className="report-library">
            <OriginalReportReader />
            <article className="extension-report-card"><span className="report-status extension">Independent extension · 2026</span><h3>Representation Drift Lab</h3><p>A generated evidence report backed by the same checksummed artifacts used by this website. It explains the code audit, corrected methods, nine interventions, three domain pairs, failure cases, reproducibility system, limitations, and next research gates.</p><a className="button primary" href={staticAsset("report/representation-drift-lab-report.pdf")} target="_blank" rel="noreferrer">Open extension report</a></article>
          </div>
          <div className="then-now"><div><span>Original report</span><strong>Asked whether drift could reveal forgetting.</strong><p>One backbone, one main dataset pair, one LoRA setup, and stronger claims from a smaller evidence base.</p></div><i aria-hidden="true">→</i><div><span>Independent extension</span><strong>Asked where that story fails.</strong><p>Rebuilt provenance, corrected supervision and geometry, three seeds, nine interventions, three pairs, negative results, tests, and an interactive explanation.</p></div></div>
        </section>
      </main>

      <footer><div><strong>Representation Drift Lab</strong><span>Ali Hasan · 2026</span></div><p>Core interactions and guided explanations work without a GPU, account, upload, or API key.</p><span className="service-status"><i /> Interactive guide · no sign-in required</span></footer>
    </>
  );
}

export default App;
