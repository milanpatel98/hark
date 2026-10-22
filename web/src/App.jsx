import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Memory } from "./engine/adapt.js";

// chosen on the two development speakers (results/tuning.json), and the same settings the Results screen reports
const BONUS = 3.0;      // phrase-list strength, per word piece
const MIN_PHRASE = 2;   // a word must be missed this many times before Hark listens for it
const MIN_RULE = 2;     // a mistake must be corrected this many times before Hark rewrites it
const MAX_ZIPF = 4.0;   // only words rarer than this (wordfreq Zipf) are listened for: names, places, jargon
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode: keep it in memory only */ } },
};
const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const lc = (t) => t.toLowerCase().replace(/(^|[.?!]\s+)([a-z])/g, (m) => m.toUpperCase()).replace(/\bi\b/g, "I");

/* ---- icons: one stroke family, square caps ---------------------------------------------------------- */
const Icon = ({ d, size = 22, fill = "none" }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={fill} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{d}</svg>
);
const I = {
  home: <path d="M4 11l8-7 8 7v9H4z" />,
  list: <path d="M5 6h14M5 12h14M5 18h9" />,
  you: <><circle cx="12" cy="8" r="4" /><path d="M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6" /></>,
  bars: <path d="M5 20V10M12 20V4M19 20v-7" />,
  mic: <><rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3" /></>,
  stop: <rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor" />,
  lock: <><rect x="5" y="11" width="14" height="10" rx="2" /><path d="M8 11V8a4 4 0 0 1 8 0v3" /></>,
  plane: <path d="M10 14l-6-2 1-2 6 1 4-7h2l-2 8 5 1 1-2h1v5h-1l-1-2-5 1 2 8h-2l-4-7z" />,
  spark: <path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6" />,
  text: <path d="M4 18 9 6l5 12M5.5 14h7M15 18l3-7 3 7M16 16h4" />,
  pen: <path d="M4 20h4L19 9l-4-4L4 16zM13 7l4 4" />,
  check: <path d="M5 12l5 5L19 7" />,
  play: <path d="M8 5v14l11-7z" fill="currentColor" />,
  save: <path d="M12 4v11M7 10l5 5 5-5M5 20h14" />,
  back: <path d="M15 5l-7 7 7 7" />,
  trash: <path d="M5 7h14M10 7V4h4v3M7 7l1 13h8l1-13" />,
  search: <><circle cx="11" cy="11" r="6" /><path d="M20 20l-4.5-4.5" /></>,
  x: <path d="M6 6l12 12M18 6 6 18" />,
};

/* ---- routing ----------------------------------------------------------------------------------------- */
function useRoute() {
  const read = () => (location.hash.replace(/^#\/?/, "") || "").split("/");
  const [r, setR] = useState(read);
  useEffect(() => {
    const on = () => { setR(read()); scrollTo(0, 0); };
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);
  return r;
}

/* ---- the engine, in a worker ------------------------------------------------------------------------- */
function useEngine(memory) {
  const worker = useRef(null);
  const [state, setState] = useState("idle"); // idle | loading | ready | listening
  const [progress, setProgress] = useState(0);
  const [lines, setLines] = useState([]);
  const [partial, setPartial] = useState("");
  const [stats, setStats] = useState(null);
  const handlers = useRef({});

  const ensure = useCallback(async () => {
    if (worker.current) return worker.current.ready;
    const w = new Worker(new URL("./engine/worker.js", import.meta.url), { type: "module" });
    worker.current = w;
    setState("loading");
    const base = new URL(".", location.href).href;
    // fetch the model once with progress so the first tap shows something happening; the browser cache serves the worker
    const sizes = { encoder: 42.8e6, decoder: 0.54e6, joiner: 0.26e6 };
    let got = 0;
    await Promise.all(Object.keys(sizes).map(async (n) => {
      const res = await fetch(`${base}model/${n}.onnx`);
      const reader = res.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        got += value.length;
        setProgress(Math.min(0.99, got / 43.6e6));
      }
    }));
    w.ready = new Promise((resolve) => {
      w.onmessage = ({ data }) => {
        if (data.type === "ready") { setState("ready"); setProgress(1); resolve(); }
        if (data.type === "text") { setPartial(data.text); setStats({ audioSec: data.audioSec, busyMs: data.busyMs }); }
        if (data.type === "line") { setPartial(""); handlers.current.line?.(data); }
        if (data.type === "done") { setStats(data); handlers.current.done?.(data); }
      };
    });
    w.postMessage({ type: "load", base });
    return w.ready;
  }, []);

  useEffect(() => {
    if (worker.current && state !== "idle" && state !== "loading") {
      worker.current.postMessage({ type: "bias", words: memory.phrases(), bonus: BONUS });
    }
  }, [memory, state]);

  const start = useCallback(async (onLine, onDone) => {
    await ensure();
    handlers.current = { line: onLine, done: onDone };
    worker.current.postMessage({ type: "bias", words: memory.phrases(), bonus: BONUS });
    worker.current.postMessage({ type: "start" });
    setPartial(""); setStats(null); setState("listening");
  }, [ensure, memory]);
  const push = useCallback((samples) => worker.current?.postMessage({ type: "audio", samples }, [samples.buffer]), []);
  const stop = useCallback(() => { worker.current?.postMessage({ type: "stop" }); setState("ready"); }, []);

  return { state, progress, lines, setLines, partial, stats, ensure, start, push, stop };
}

/* ---- audio in: microphone or a sample file ----------------------------------------------------------- */
function resampleTo16k(x, rate) {
  if (rate === 16000) return x;
  const ratio = rate / 16000, n = Math.floor(x.length / ratio), out = new Float32Array(n);
  // average over each output step: a simple low-pass that keeps speech clean when the device runs at 44.1 or 48 kHz
  for (let i = 0; i < n; i++) {
    const a = i * ratio, b = a + ratio;
    let s = 0, c = 0;
    for (let j = Math.floor(a); j < Math.min(x.length, Math.ceil(b)); j++) { s += x[j]; c++; }
    out[i] = c ? s / c : 0;
  }
  return out;
}

async function openMic(onAudio, onLevel) {
  const media = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 } });
  let ctx;
  try { ctx = new AudioContext({ sampleRate: 16000 }); } catch { ctx = new AudioContext(); }
  await ctx.audioWorklet.addModule(new URL("mic-worklet.js", location.href).href);
  const src = ctx.createMediaStreamSource(media);
  const tap = new AudioWorkletNode(ctx, "tap");
  tap.port.onmessage = ({ data }) => {
    let e = 0;
    for (let i = 0; i < data.length; i++) e += data[i] * data[i];
    onLevel(Math.min(1, Math.sqrt(e / data.length) * 6));
    onAudio(resampleTo16k(data, ctx.sampleRate));
  };
  src.connect(tap);
  return () => { media.getTracks().forEach((t) => t.stop()); ctx.close(); };
}

async function playSample(url, onAudio, onLevel, signal) {
  const buf = new Uint8Array(await (await fetch(url)).arrayBuffer());
  const dv = new DataView(buf.buffer);
  let p = 12, data = null;
  while (p < buf.length) {
    const id = String.fromCharCode(...buf.subarray(p, p + 4)), size = dv.getUint32(p + 4, true);
    if (id === "data") data = new Int16Array(buf.buffer.slice(p + 8, p + 8 + size));
    p += 8 + size + (size & 1);
  }
  const wave = Float32Array.from(data, (v) => v / 32768);
  const audio = new Audio(url);
  audio.play().catch(() => {});
  for (let i = 0; i < wave.length && !signal.aborted; i += 1600) {
    const chunk = wave.slice(i, i + 1600);
    let e = 0;
    for (const v of chunk) e += v * v;
    onLevel(Math.min(1, Math.sqrt(e / chunk.length) * 6));
    onAudio(chunk);
    await new Promise((r) => setTimeout(r, 100));
  }
  audio.pause();
}

/* ---- pieces ------------------------------------------------------------------------------------------ */
function Dock({ page }) {
  const items = [["", "Home", I.home], ["saved", "Saved", I.list], ["you", "Your voice", I.you], ["results", "Results", I.bars]];
  return (
    <nav className="dock" aria-label="Main">
      {items.map(([k, label, icon]) => (
        <a key={k} href={`#/${k}`} className={page === k ? "on" : ""} aria-label={label} aria-current={page === k ? "page" : undefined}>
          <Icon d={icon} size={21} />
        </a>
      ))}
    </nav>
  );
}

function Top({ title, back, right }) {
  return (
    <header className="top">
      {back ? <a href={back} className="round small" aria-label="Back"><Icon d={I.back} /></a> : <img src="./brand/icon-white.png" alt="Hark" className="mark" />}
      {title && <h1>{title}</h1>}
      <span className="top-right">{right}</span>
    </header>
  );
}

function Wave({ levels, tone = "soft" }) {
  return (
    <div className={`wave ${tone}`} aria-hidden="true">
      {levels.map((v, i) => <i key={i} style={{ height: `${Math.max(4, v * 40)}px` }} />)}
    </div>
  );
}

function Ring({ progress }) {
  const r = 70, c = 2 * Math.PI * r;
  return (
    <svg className="ring" viewBox="0 0 160 160" aria-hidden="true">
      <circle cx="80" cy="80" r={r} className="ring-track" />
      <circle cx="80" cy="80" r={r} className="ring-fill" strokeDasharray={c} strokeDashoffset={c * (1 - progress)} />
    </svg>
  );
}

function Toast({ text }) {
  return text ? <div className="toast" role="status"><Icon d={I.spark} size={18} />{text}</div> : null;
}

/* ---- home -------------------------------------------------------------------------------------------- */
function Home({ engine, onStart }) {
  const loading = engine.state === "loading";
  return (
    <section className="screen home">
      <header className="top"><img src="./brand/wordmark.png" alt="Hark" className="wordmark" /></header>
      <div className="home-center">
        <button className={`mic ${loading ? "loading" : ""}`} onClick={() => onStart("mic")} aria-label="Start live captions">
          {loading && <Ring progress={engine.progress} />}
          <span className="mic-core"><Icon d={I.mic} size={32} /></span>
        </button>
        <p className="hint">{loading ? `${Math.round(engine.progress * 100)}%` : "Tap to caption"}</p>
        <div className="chips">
          <span className="chip" title="Audio never leaves this device"><Icon d={I.lock} /><small>Private</small></span>
          <span className="chip" title="Works offline once loaded"><Icon d={I.plane} /><small>Offline</small></span>
          <span className="chip" title="Learns your words from your fixes"><Icon d={I.spark} /><small>Learns</small></span>
        </div>
        <button className="pill ghost" onClick={() => onStart("sample")}><Icon d={I.play} size={14} />Try a sample</button>
      </div>
    </section>
  );
}

/* ---- live -------------------------------------------------------------------------------------------- */
function Line({ line, onFix }) {
  const [editing, setEditing] = useState(false);
  const [val, setVal] = useState(line.fixed || line.shown);
  if (editing)
    return (
      <form className="line editing" onSubmit={(e) => { e.preventDefault(); onFix(val); setEditing(false); }}>
        <textarea value={val} onChange={(e) => setVal(e.target.value)} rows={3} autoFocus aria-label="Correct this caption" />
        <div className="line-actions">
          <button type="button" className="round small" onClick={() => setEditing(false)} aria-label="Cancel"><Icon d={I.x} /></button>
          <button type="submit" className="round small yellow" aria-label="Save and learn"><Icon d={I.check} /></button>
        </div>
      </form>
    );
  return (
    <p className={`line ${line.fixed ? "fixed" : ""}`}>
      {lc(line.fixed || line.shown)}
      <button className="fix" onClick={() => setEditing(true)} aria-label="Fix this line">
        <Icon d={line.fixed ? I.check : I.pen} size={16} />
      </button>
    </p>
  );
}

function Live({ engine, lines, partial, size, setSize, listening, onStop, onFix, elapsed, levels, source, learned }) {
  const box = useRef(null);
  useEffect(() => { box.current?.scrollTo({ top: box.current.scrollHeight, behavior: "smooth" }); }, [lines.length, partial]);
  return (
    <section className="screen live">
      <Top back="#/" right={<span className={`rec ${listening ? "on" : ""}`}><i />{fmtTime(elapsed)}</span>} />
      <div className={`captions s${size}`} ref={box} aria-live="polite">
        {!lines.length && !partial && <p className="empty">{engine.state === "loading" ? "Loading…" : listening ? "Listening…" : "Nothing yet"}</p>}
        {lines.map((l, i) => <Line key={l.id} line={l} onFix={(t) => onFix(i, t)} />)}
        {partial && <p className="line partial">{lc(partial)}</p>}
      </div>
      <Wave levels={levels} tone={listening ? "live" : "soft"} />
      <div className="controls">
        <button className="round" onClick={() => setSize((s) => (s % 3) + 1)} aria-label="Text size"><Icon d={I.text} /></button>
        {listening
          ? <button className="round big yellow" onClick={onStop} aria-label="Stop"><Icon d={I.stop} size={28} /></button>
          : <a className="round big yellow" href="#/" aria-label="Done"><Icon d={I.check} size={28} /></a>}
        <a className="round learned" href="#/you" aria-label={`${learned} words learned`}><Icon d={I.spark} />{learned > 0 && <em>{learned}</em>}</a>
      </div>
      {source === "sample" && <p className="credit">Sample: L2-ARCTIC speaker SKA, Arabic accent (CC BY-NC 4.0)</p>}
    </section>
  );
}

/* ---- saved ------------------------------------------------------------------------------------------- */
function Saved({ sessions }) {
  const [q, setQ] = useState("");
  const shown = sessions.filter((s) => !q || s.lines.some((l) => (l.fixed || l.shown).toLowerCase().includes(q.toLowerCase())));
  return (
    <section className="screen">
      <Top title="Saved" />
      <label className="search"><Icon d={I.search} size={18} /><input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search" aria-label="Search transcripts" /></label>
      {!sessions.length && <div className="blank"><Icon d={I.list} size={34} /><p>Your captions land here</p></div>}
      {shown.map((s) => (
        <a key={s.id} className="card" href={`#/saved/${s.id}`}>
          <Wave levels={s.shape} tone="yellow" />
          <span className="card-text"><b>{s.title}</b><small>{new Date(s.date).toLocaleDateString(undefined, { month: "short", day: "numeric" })}</small></span>
          <small className="dur">{fmtTime(s.duration)}</small>
        </a>
      ))}
    </section>
  );
}

function Session({ s, onFix, onDelete }) {
  if (!s) return <section className="screen"><Top back="#/saved" /><div className="blank"><p>Not found</p></div></section>;
  return (
    <section className="screen">
      <Top back="#/saved" title={s.title} right={<button className="round small" onClick={onDelete} aria-label="Delete"><Icon d={I.trash} /></button>} />
      <div className="captions s1 page">
        {s.lines.map((l, i) => <Line key={l.id} line={l} onFix={(t) => onFix(s.id, i, t)} />)}
      </div>
    </section>
  );
}

/* ---- your voice -------------------------------------------------------------------------------------- */
function You({ memory, onReset }) {
  const words = memory.phrases(60);
  const pending = memory.pending().slice(0, 40);
  const rules = [...memory.activeRules()];
  return (
    <section className="screen">
      <Top title="Your voice" />
      <div className="stat-row">
        <div className="stat"><b>{memory.corrections}</b><small>{memory.corrections === 1 ? "fix" : "fixes"}</small></div>
        <div className="stat"><b>{memory.phrases().length}</b><small>{memory.phrases().length === 1 ? "word" : "words"} learned</small></div>
        <div className="stat"><b>{rules.length}</b><small>auto-{rules.length === 1 ? "fix" : "fixes"}</small></div>
      </div>
      {words.length ? (
        <>
          <h2 className="label"><Icon d={I.spark} size={16} />Listens harder for</h2>
          <div className="tags">{words.map((w) => <span key={w} className="tag">{w.toLowerCase()}</span>)}</div>
        </>
      ) : !pending.length && (
        <div className="blank"><Icon d={I.pen} size={34} /><p>Fix a caption and Hark learns from it</p></div>
      )}
      {pending.length > 0 && (
        <>
          <h2 className="label"><Icon d={I.pen} size={16} />Seen once · one more fix to learn</h2>
          <div className="tags">{pending.map((w) => <span key={w} className="tag faint">{w.toLowerCase()}</span>)}</div>
        </>
      )}
      {rules.length > 0 && (
        <>
          <h2 className="label"><Icon d={I.check} size={16} />Fixes on its own</h2>
          <ul className="rules">{rules.map(([a, b]) => <li key={a}><s>{a.toLowerCase()}</s><Icon d={<path d="M5 12h14M13 6l6 6-6 6" />} size={16} /><b>{b.toLowerCase()}</b></li>)}</ul>
        </>
      )}
      <p className="note"><Icon d={I.lock} size={16} />Stored only in this browser</p>
      {memory.corrections > 0 && <button className="pill ghost danger" onClick={onReset}><Icon d={I.trash} size={18} />Forget my voice</button>}
    </section>
  );
}

/* ---- results ----------------------------------------------------------------------------------------- */
function Results({ data }) {
  if (!data) return <section className="screen"><Top title="Results" /><div className="blank"><p>Loading…</p></div></section>;
  if (!data.speakers) return <section className="screen"><Top title="Results" /><div className="blank"><Icon d={I.bars} size={34} /><p>Benchmark in progress</p></div></section>;
  const { summary: t, speakers, first_run: f } = data;
  const pc = (e, n) => `${((e / n) * 100).toFixed(1)}%`;
  const max = Math.max(...speakers.map((s) => s.before));
  return (
    <section className="screen">
      <Top title="Results" />
      <div className="hero-num">
        <b>{t.learned_wrong_before}→{t.learned_wrong_after}</b>
        <small>misses on names you had corrected</small>
      </div>
      <div className="duo">
        <div><b>{pc(t.errors_before, t.words)}</b><small>words wrong, before</small></div>
        <div><b>{pc(t.errors_after, t.words)}</b><small>after {t.fixes} fixes</small></div>
        <div><b>{t.speakers}</b><small>accented speakers, held out</small></div>
      </div>
      <p className="note"><Icon d={I.check} size={16} />Never worse overall · learns only rare words</p>
      <h2 className="label"><Icon d={I.you} size={16} />Each speaker, words wrong before → after</h2>
      <div className="bars">
        {speakers.map((s) => (
          <div key={s.speaker} className={`bar-row ${s.split}`}>
            <span className="who"><b>{s.accent}</b><small>{s.sex === "F" ? "♀" : "♂"}{s.split === "dev" ? " · tuning" : ""}</small></span>
            <span className="bar-track">
              <i className="before" style={{ width: `${(s.before / max) * 100}%` }} />
              <i className="after" style={{ width: `${(s.after / max) * 100}%` }} />
            </span>
            <span className="num">{(s.before * 100).toFixed(1)}→{(s.after * 100).toFixed(1)}</span>
          </div>
        ))}
      </div>
      {f && (
        <p className="note"><Icon d={I.spark} size={16} />First run listened for common words too: {pc(f.errors_before, f.words)} → {pc(f.errors_after, f.words)} overall. Fixed.</p>
      )}
      <p className="credit">
        Speech: <a href="https://psi.engr.tamu.edu/l2-arctic-corpus/">L2-ARCTIC</a> (CC BY-NC 4.0). Model: k2-fsa Zipformer (Apache-2.0).
      </p>
    </section>
  );
}

/* ---- app --------------------------------------------------------------------------------------------- */
export default function App() {
  const route = useRoute();
  const page = route[0] || "";
  const [memJSON, setMemJSON] = useState(() => store.get("hark.memory", null));
  const [common, setCommon] = useState(null);
  useEffect(() => { fetch("./data/common.json").then((r) => r.json()).then(setCommon).catch(() => setCommon({})); }, []);
  const memory = useMemo(() => {
    const opts = { minRuleCount: MIN_RULE, minPhraseCount: MIN_PHRASE, maxZipf: MAX_ZIPF, common: common || {} };
    return memJSON ? Memory.fromJSON(memJSON, opts) : new Memory(opts);
  }, [memJSON, common]);
  const engine = useEngine(memory);
  const [lines, setLines] = useState([]);
  const [listening, setListening] = useState(false);
  const [size, setSize] = useState(() => store.get("hark.size", 2));
  const [levels, setLevels] = useState(() => Array(24).fill(0.08));
  const [elapsed, setElapsed] = useState(0);
  const [source, setSource] = useState("mic");
  const [sessions, setSessions] = useState(() => store.get("hark.sessions", []));
  const [toast, setToast] = useState("");
  const [results, setResults] = useState(null);
  const stopRef = useRef(null);
  const shapeRef = useRef([]);
  const linesRef = useRef([]);
  useEffect(() => { linesRef.current = lines; }, [lines]);
  useEffect(() => store.set("hark.size", size), [size]);
  useEffect(() => { fetch("./data/results.json").then((r) => r.json()).then(setResults).catch(() => setResults({})); }, []);
  useEffect(() => { if (toast) { const t = setTimeout(() => setToast(""), 2600); return () => clearTimeout(t); } }, [toast]);

  const learn = useCallback((heard, said) => {
    const before = new Set(memory.phrases()), beforePending = new Set(memory.pending());
    memory.learn(heard, said);
    const json = memory.toJSON();
    store.set("hark.memory", json);
    setMemJSON(json);
    const fresh = memory.phrases().filter((w) => !before.has(w));
    const noted = memory.pending().filter((w) => !beforePending.has(w));
    setToast(fresh.length ? `Learned ${fresh.slice(0, 3).map((w) => w.toLowerCase()).join(", ")}`
      : noted.length ? `Noted ${noted.slice(0, 2).map((w) => w.toLowerCase()).join(", ")}. Learns after one more fix` : "Thanks, noted");
  }, [memory]);

  const finishSession = useCallback(() => {
    const ls = linesRef.current;
    if (!ls.length) return;
    const raw = shapeRef.current, shape = Array.from({ length: 24 }, (_, i) => raw[Math.floor((i * raw.length) / 24)] ?? 0.05);
    const s = { id: Date.now().toString(36), date: Date.now(), duration: elapsed, title: lc(ls[0].fixed || ls[0].shown).split(" ").slice(0, 4).join(" "), lines: ls, shape };
    setSessions((all) => { const next = [s, ...all].slice(0, 50); store.set("hark.sessions", next); return next; });
  }, [elapsed]);

  const start = useCallback(async (kind) => {
    setSource(kind); setLines([]); shapeRef.current = []; setElapsed(0);
    location.hash = "#/live";
    const t0 = Date.now();
    const timer = setInterval(() => setElapsed((Date.now() - t0) / 1000), 250);
    const onLine = (d) => setLines((ls) => [...ls, { id: `${t0}-${d.n}`, heard: d.text, shown: memory.rewrite(d.text) }]);
    const onDone = () => { clearInterval(timer); setListening(false); setTimeout(finishSession, 50); };
    await engine.start(onLine, onDone);
    setListening(true);
    const onLevel = (v) => { shapeRef.current.push(v); setLevels((l) => [...l.slice(1), v]); };
    if (kind === "mic") {
      try {
        const close = await openMic(engine.push, onLevel);
        stopRef.current = () => { close(); engine.stop(); };
      } catch {
        setToast("Microphone blocked. Allow it, or try the sample");
        clearInterval(timer); setListening(false); engine.stop();
      }
    } else {
      const ctl = new AbortController();
      stopRef.current = () => { ctl.abort(); engine.stop(); };
      await playSample("./samples/l2arctic-ska-a0049.wav", engine.push, onLevel, ctl.signal);
      if (!ctl.signal.aborted) engine.stop();
    }
  }, [engine, memory, finishSession]);

  const stop = useCallback(() => { stopRef.current?.(); stopRef.current = null; }, []);

  const fixLive = (i, text) => {
    const l = lines[i];
    learn(l.heard, text);
    setLines((ls) => ls.map((x, j) => (j === i ? { ...x, fixed: text } : x)));
    setSessions((all) => { const next = all.map((s) => ({ ...s, lines: s.lines.map((x) => (x.id === l.id ? { ...x, fixed: text } : x)) })); store.set("hark.sessions", next); return next; });
  };
  const fixSaved = (sid, i, text) => {
    const s = sessions.find((x) => x.id === sid);
    learn(s.lines[i].heard, text);
    setSessions((all) => { const next = all.map((x) => (x.id === sid ? { ...x, lines: x.lines.map((l, j) => (j === i ? { ...l, fixed: text } : l)) } : x)); store.set("hark.sessions", next); return next; });
  };

  let body;
  if (page === "live") body = <Live engine={engine} lines={lines} partial={engine.partial} size={size} setSize={setSize} listening={listening} onStop={stop} onFix={fixLive} elapsed={elapsed} levels={levels} source={source} learned={memory.phrases().length} />;
  else if (page === "saved" && route[1]) body = <Session s={sessions.find((x) => x.id === route[1])} onFix={fixSaved} onDelete={() => { const next = sessions.filter((x) => x.id !== route[1]); setSessions(next); store.set("hark.sessions", next); location.hash = "#/saved"; }} />;
  else if (page === "saved") body = <Saved sessions={sessions} />;
  else if (page === "you") body = <You memory={memory} onReset={() => { store.set("hark.memory", null); setMemJSON(null); setToast("Forgotten"); }} />;
  else if (page === "results") body = <Results data={results} />;
  else body = <Home engine={engine} onStart={start} />;

  return (
    <div className="app">
      {body}
      <Toast text={toast} />
      {page !== "live" && <Dock page={page} />}
      <footer className="foot">© 2026 Milan Patel · MIT License</footer>
    </div>
  );
}
