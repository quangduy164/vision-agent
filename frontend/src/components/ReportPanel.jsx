import React from 'react';

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

const SAFETY_EN = 'This is an AI-assisted analysis';
const SAFETY_VI = 'Đây là phân tích hỗ trợ bởi AI';

function highlightKeywords(text) {
  const escaped = KEYWORDS.map(k => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  const regex   = new RegExp(`(${escaped.join('|')})`, 'gi');
  const parts   = text.split(regex);
  return parts.map((part, i) =>
    regex.test(part)
      ? <strong key={i} className="report-finding">{part}</strong>
      : part
  );
}

function highlightLine(text) {
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
}

export default function ReportPanel({ report, t }) {
  const lines = report.split('\n').filter(Boolean);
  const body  = lines.filter(l =>
    !l.startsWith(SAFETY_EN) && !l.startsWith(SAFETY_VI) && !l.startsWith('Cần đối chiếu')
  );
  const disc = lines.find(l =>
    l.startsWith(SAFETY_EN) || l.startsWith(SAFETY_VI) || l.startsWith('Cần đối chiếu')
  );

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
