import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import translations from './i18n';
import flagEN from './assets/english.png';
import flagVI from './assets/vietnam.png';
import logo   from './assets/logo.png';
import UploadCard   from './components/UploadCard';
import AnalysisCard from './components/AnalysisCard';
import ReportPanel  from './components/ReportPanel';

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

  useEffect(() => {
    if (result) setReport(result.report);
  }, [result]);

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

  const uploadProps = {
    t, preview, file, loading, error, inputRef,
    onFileChange: (e) => handleFile(e.target.files[0]),
    onDrop:       (e) => { e.preventDefault(); handleFile(e.dataTransfer.files[0]); },
    onAnalyze:    handleAnalyze,
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-brand">
          <img src={logo} alt="MedAgent AI" className="header-logo" />
          <div className="header-text">
            <h1>MedAgent AI</h1>
            <p>{t.appSubtitle}</p>
          </div>
        </div>
        <button
          className={`lang-toggle ${lang === 'vi' ? 'lang-vi' : 'lang-en'}`}
          onClick={() => setLang(l => l === 'en' ? 'vi' : 'en')}
          aria-label="Switch language"
        >
          <span className="lang-knob">
            <img src={lang === 'en' ? flagEN : flagVI} alt={lang} className="lang-flag" />
          </span>
          <span className="lang-label">{lang === 'en' ? 'EN' : 'VI'}</span>
        </button>
      </header>

      <div className="content">
        {!result && (
          <div className="layout-single">
            <UploadCard {...uploadProps} />
          </div>
        )}

        {result && !isAbnormal && (
          <div className="layout-two layout-two--equal">
            <UploadCard {...uploadProps} />
            <AnalysisCard result={result} preview={null} heatmapUrl={null} t={t} />
          </div>
        )}

        {result && isAbnormal && (
          <div className="layout-two">
            <div className="left-col">
              <UploadCard {...uploadProps} />
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
