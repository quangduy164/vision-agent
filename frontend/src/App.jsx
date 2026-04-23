import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import translations from './i18n';
import flagEN from './assets/english.png';
import flagVI from './assets/vietnam.png';
import logo        from './assets/logo.png';
import analyzeIcon from './assets/analyze_icon.png';

const API      = '/analyze-image';
const IMG_BASE = 'http://localhost:8000';

export default function App() {
  const [file, setFile]       = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const [lang, setLang]       = useState('en');
  const [report, setReport]   = useState(null);
  const inputRef              = useRef();
  const t = translations[lang];

  // Sync report khi có result mới
  useEffect(() => {
    if (result) setReport(result.report);
  }, [result]);

  // Dịch lại report khi đổi ngôn ngữ
  useEffect(() => {
    if (!result) return;
    const findings = result.visual_findings || {};
    const form = new FormData();
    form.append('diagnosis',  result.diagnosis);
    form.append('confidence', result.confidence);
    form.append('location',   findings.location || 'chest');
    form.append('size',       findings.size     || 'moderate');
    form.append('side',       findings.side     || 'unspecified');
    form.append('lang',       lang);
    fetch('/translate-report', { method: 'POST', body: form })
      .then(r => r.json())
      .then(j => { if (j.success) setReport(j.report); })
      .catch(() => {});
  }, [lang]); // eslint-disable-line

  const handleFile = (f) => {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setReport(null);
    setError(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    handleFile(e.dataTransfer.files[0]);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append('file', file);
      const res  = await fetch(`${API}?lang=${lang}`, { method: 'POST', body: form });
      const json = await res.json();
      if (!json.success) throw new Error(json.error || 'Server error');
      setResult(json.data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const heatmapUrl = result?.heatmap_path
    ? `${IMG_BASE}/outputs/${result.heatmap_path.replace(/\\/g, '/').split('/').pop()}`
    : null;

  const isAbnormal = result && result.status !== 'Normal';

  const UploadCard = (
    <section className="card upload-card">
      <h2>{t.uploadTitle}</h2>
      <div
        className={`dropzone ${preview ? 'has-image' : ''}`}
        onClick={() => inputRef.current.click()}
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
      >
        {preview
          ? <img src={preview} alt="preview" className="preview-img" />
          : <div className="dropzone-hint">
              <span className="drop-icon">📂</span>
              <p>{t.uploadHint} <u>{t.uploadClick}</u></p>
              <p className="hint-sub">{t.uploadSub}</p>
            </div>
        }
      </div>
      <input ref={inputRef} type="file" accept="image/*"
        style={{ display: 'none' }}
        onChange={(e) => handleFile(e.target.files[0])} />
      {file && <p className="filename">📄 {file.name}</p>}
      <button className="btn-analyze" onClick={handleAnalyze} disabled={!file || loading}>
        {loading
          ? <><span className="spinner" /> {t.analyzing}</>
          : <>{t.analyzeBtn} <img src={analyzeIcon} alt="" className="btn-icon" /></>
        }
      </button>
      {error && <p className="error-msg">❌ {error}</p>}
    </section>
  );

  return (
    <div className="app">
      <header className="header">
        <div className="header-brand">
          <img src={logo} alt="MedAgent AI logo" className="header-logo" />
          <div className="header-text">
            <h1>MedAgent AI</h1>
            <p>{t.appSubtitle}</p>
          </div>
        </div>
        <button
          className={`lang-toggle ${lang === 'vi' ? 'lang-vi' : 'lang-en'}`}
          onClick={() => setLang(l => l === 'en' ? 'vi' : 'en')}
          title="Switch language"
          aria-label="Switch language"
        >
          <span className="lang-knob">
            <img src={lang === 'en' ? flagEN : flagVI} alt={lang} className="lang-flag" />
          </span>
          <span className="lang-label">
            {lang === 'en' ? 'EN' : 'VI'}
          </span>
        </button>
      </header>

      <div className="content">

      {!result && (
        <div className="layout-single">{UploadCard}</div>
      )}

      {result && !isAbnormal && (
        <div className="layout-two layout-two--equal">
          {UploadCard}
          <AnalysisCard result={result} preview={null} heatmapUrl={null} t={t} />
        </div>
      )}

      {result && isAbnormal && (
        <div className="layout-two">
          <div className="left-col">
            {UploadCard}
            <ReportPanel report={report || result.report} t={t} />
          </div>
          <AnalysisCard result={result} preview={preview} heatmapUrl={heatmapUrl} stretch t={t} />
        </div>
      )}

      {result && !isAbnormal && (
        <div style={{ marginTop: 24 }}>
          <ReportPanel report={report || result.report} t={t} />
        </div>
      )}
      </div>
    </div>
  );
}

/* ── Analysis Card ── */
function AnalysisCard({ result, preview, heatmapUrl, stretch, t }) {
  const isNormal   = result.status === 'Normal';
  const confidence = (result.confidence * 100).toFixed(1);
  const top5 = Object.entries(result.all_probabilities)
    .sort((a, b) => b[1] - a[1]).slice(0, 5);

  return (
    <section className={`card result-card ${stretch ? 'stretch' : ''}`}>
      <h2>{t.analysisTitle}</h2>

      <div className={`status-badge ${isNormal ? 'normal' : 'abnormal'}`}>
        {isNormal ? t.statusNormal : t.statusAbnormal}
      </div>

      {!isNormal && heatmapUrl && (
        <div className="heatmap-section">
          <h3>{t.heatmapTitle}</h3>
          <div className="heatmap-compare">
            <div className="heatmap-item">
              <span className="heatmap-label">{t.heatmapOriginal}</span>
              <img src={preview} alt="original" />
            </div>
            <div className="heatmap-item">
              <span className="heatmap-label">{t.heatmapActivation}</span>
              <img src={heatmapUrl} alt="heatmap" />
            </div>
          </div>
        </div>
      )}

      <div className="info-grid">
        <InfoItem label={t.labelDiagnosis}  value={t.diseases[result.diagnosis] || result.diagnosis} />
        <InfoItem label={t.labelConfidence} value={`${confidence}%`} />
      </div>

      <div className="prob-section">
        <h3>{t.topProb}</h3>
        {top5.map(([d, p]) => <ProbBar key={d} disease={d} prob={p} t={t} />)}
      </div>
    </section>
  );
}

/* ── Report Panel ── */
function ReportPanel({ report, t }) {
  const SAFETY_EN = 'This is an AI-assisted analysis';
  const SAFETY_VI = 'Đây là phân tích hỗ trợ bởi AI';

  const lines = report.split('\n').filter(Boolean);
  const body  = lines.filter(l => !l.startsWith(SAFETY_EN) && !l.startsWith(SAFETY_VI) && !l.startsWith('Cần đối chiếu'));
  const disc  = lines.find(l => l.startsWith(SAFETY_EN) || l.startsWith(SAFETY_VI) || l.startsWith('Cần đối chiếu'));

  const KEYWORDS = [
    'Findings:', 'Impression:', 'No Finding', 'Cardiomegaly', 'Atelectasis',
    'Effusion', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Nodule', 'Mass', 'Emphysema', 'Fibrosis', 'Pleural_Thickening',
    'Infiltration', 'Lung Opacity', 'Fracture', 'Hernia',
    'No acute', 'Clinical correlation', 'CT recommended', 'Follow-up',
    'Kết quả:', 'Kết luận:', 'Không phát hiện bất thường',
    'Tim to', 'Xẹp phổi', 'Đông đặc phổi', 'Phù phổi', 'Tràn dịch màng phổi',
    'Khí phế thũng', 'Xơ phổi', 'Gãy xương sườn', 'Thoát vị hoành',
    'Thâm nhiễm phổi', 'Mờ phổi', 'Khối u phổi', 'Nốt phổi',
    'Dày màng phổi', 'Viêm phổi', 'Tràn khí màng phổi',
    'Cần đối chiếu lâm sàng', 'Cần chụp CT', 'Cần theo dõi',
  ];

  const highlightLine = (text) => {
    const labelMatch = text.match(/^(Findings:|Impression:|Kết quả:|Kết luận:)(.*)/s);
    if (labelMatch) {
      return (
        <>
          <span className="report-label">{labelMatch[1]}</span>
          {highlightKeywords(labelMatch[2])}
        </>
      );
    }
    return highlightKeywords(text);
  };

  const highlightKeywords = (text) => {
    const escaped = KEYWORDS.map(k => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    const regex   = new RegExp(`(${escaped.join('|')})`, 'gi');
    const parts   = text.split(regex);
    return parts.map((part, i) =>
      regex.test(part)
        ? <strong key={i} className="report-finding">{part}</strong>
        : part
    );
  };

  return (
    <section className="card report-card">
      <h2>{t.reportTitle}</h2>
      <div className="report-body">
        {body.map((line, i) => <p key={i}>{highlightLine(line)}</p>)}
      </div>
      {disc && <p className="disclaimer">{disc}</p>}
    </section>
  );
}

function InfoItem({ label, value }) {
  return (
    <div className="info-item">
      <span className="info-label">{label}</span>
      <span className="info-value">{value}</span>
    </div>
  );
}

function ProbBar({ disease, prob, t }) {
  const displayName = t?.diseases?.[disease] || disease;
  return (
    <div className="prob-row">
      <span className="prob-name" title={disease}>{displayName}</span>
      <div className="prob-track">
        <div className="prob-fill" style={{ width: `${Math.min(prob * 100, 100)}%` }} />
      </div>
      <span className="prob-pct">{(prob * 100).toFixed(1)}%</span>
    </div>
  );
}
