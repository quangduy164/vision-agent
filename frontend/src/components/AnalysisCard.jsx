import React from 'react';

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

export default function AnalysisCard({ result, preview, heatmapUrl, stretch, t }) {
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
