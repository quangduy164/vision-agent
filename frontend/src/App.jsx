import React, { useState, useRef } from 'react';
import './App.css';

const API      = '/analyze-image';
const IMG_BASE = 'http://localhost:8000';

export default function App() {
  const [file, setFile]       = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const inputRef              = useRef();

  const handleFile = (f) => {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
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
      const res  = await fetch(API, { method: 'POST', body: form });
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
      <h2>Upload X-Ray Image</h2>
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
              <p>Drag & drop or <u>click to select</u></p>
              <p className="hint-sub">PNG, JPG supported</p>
            </div>
        }
      </div>
      <input ref={inputRef} type="file" accept="image/*"
        style={{ display: 'none' }}
        onChange={(e) => handleFile(e.target.files[0])} />
      {file && <p className="filename">📄 {file.name}</p>}
      <button className="btn-analyze" onClick={handleAnalyze} disabled={!file || loading}>
        {loading ? <><span className="spinner" /> Analyzing...</> : '🔍 Analyze'}
      </button>
      {error && <p className="error-msg">❌ {error}</p>}
    </section>
  );

  return (
    <div className="app">
      <header className="header">
        <span className="header-icon">🫁</span>
        <div>
          <h1>X-Ray AI Agent</h1>
          <p>Multi-Agent Chest X-Ray Analysis · DenseNet121 + ResNet50 Ensemble</p>
        </div>
      </header>

      {/* ── Chưa có kết quả: chỉ Upload ── */}
      {!result && (
        <div className="layout-single">
          {UploadCard}
        </div>
      )}

      {/* ── Normal: 2 cột bằng nhau [Upload | Analysis] ── */}
      {result && !isAbnormal && (
        <div className="layout-two layout-two--equal">
          {UploadCard}
          <AnalysisCard result={result} preview={null} heatmapUrl={null} />
        </div>
      )}

      {/* ── Abnormal: 2 cột [Upload+Report | Analysis+Heatmap] ── */}
      {result && isAbnormal && (
        <div className="layout-two">
          {/* Cột trái: Upload + Report */}
          <div className="left-col">
            {UploadCard}
            <ReportPanel report={result.report} />
          </div>
          {/* Cột phải: Analysis với heatmap, stretch bằng cột trái */}
          <AnalysisCard result={result} preview={preview} heatmapUrl={heatmapUrl} stretch />
        </div>
      )}

      {/* ── Normal: Report full-width bên dưới ── */}
      {result && !isAbnormal && (
        <div style={{ marginTop: 24 }}>
          <ReportPanel report={result.report} />
        </div>
      )}
    </div>
  );
}

/* ── Analysis Card ── */
function AnalysisCard({ result, preview, heatmapUrl, stretch }) {
  const isNormal   = result.status === 'Normal';
  const confidence = (result.confidence * 100).toFixed(1);
  const top5 = Object.entries(result.all_probabilities)
    .sort((a, b) => b[1] - a[1]).slice(0, 5);

  return (
    <section className={`card result-card ${stretch ? 'stretch' : ''}`}>
      <h2>Analysis Result</h2>

      <div className={`status-badge ${isNormal ? 'normal' : 'abnormal'}`}>
        {isNormal ? '✅ Normal' : '⚠️ Abnormal'}
      </div>

      {/* Heatmap - chỉ khi Abnormal */}
      {!isNormal && heatmapUrl && (
        <div className="heatmap-section">
          <h3>Grad-CAM Heatmap</h3>
          <div className="heatmap-compare">
            <div className="heatmap-item">
              <span className="heatmap-label">Original</span>
              <img src={preview} alt="original" />
            </div>
            <div className="heatmap-item">
              <span className="heatmap-label">Activation Map</span>
              <img src={heatmapUrl} alt="heatmap" />
            </div>
          </div>
        </div>
      )}

      {/* Info: chỉ Diagnosis + Confidence */}
      <div className="info-grid">
        <InfoItem label="Diagnosis"  value={result.diagnosis} />
        <InfoItem label="Confidence" value={`${confidence}%`} />
      </div>

      {/* Probability bars */}
      <div className="prob-section">
        <h3>Top Probabilities</h3>
        {top5.map(([d, p]) => <ProbBar key={d} disease={d} prob={p} />)}
      </div>
    </section>
  );
}

/* ── Report Panel ── */
function ReportPanel({ report }) {
  const lines = report.split('\n').filter(Boolean);
  const body  = lines.filter(l => !l.startsWith('This is'));
  const disc  = lines.find(l => l.startsWith('This is'));

  // Các từ khóa y khoa quan trọng cần in đậm
  const KEYWORDS = [
    'Findings:', 'Impression:', 'No Finding', 'Normal', 'Abnormal',
    'Cardiomegaly', 'Atelectasis', 'Effusion', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Nodule', 'Mass', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Infiltration', 'Lung Opacity', 'Fracture', 'Hernia',
    'No acute', 'No pneumothorax', 'No pleural effusion', 'No focal consolidation',
    'Clinical correlation', 'CT recommended', 'Follow-up',
  ];

  const highlightLine = (text) => {
    // Tách "Findings:" / "Impression:" thành label riêng
    const labelMatch = text.match(/^(Findings:|Impression:)(.*)/s);
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
    const regex = new RegExp(`(${KEYWORDS.map(k => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi');
    const parts = text.split(regex);
    return parts.map((part, i) =>
      regex.test(part)
        ? <strong key={i} className="report-finding">{part}</strong>
        : part
    );
  };

  return (
    <section className="card report-card">
      <h2>Radiology Report</h2>
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

function ProbBar({ disease, prob }) {
  return (
    <div className="prob-row">
      <span className="prob-name">{disease}</span>
      <div className="prob-track">
        <div className="prob-fill" style={{ width: `${Math.min(prob * 100, 100)}%` }} />
      </div>
      <span className="prob-pct">{(prob * 100).toFixed(1)}%</span>
    </div>
  );
}
