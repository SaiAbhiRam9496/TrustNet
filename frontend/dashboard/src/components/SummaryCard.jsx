import React from 'react';
import { AlertCircle, CheckCircle2, HelpCircle, XCircle } from 'lucide-react';

export default function SummaryCard({ data }) {
  if (!data) return null;

  const { verdict_label, confidence, overall_score, is_uncertain, uncertainty_reason, summary, factors } = data;

  const renderVerdictIcon = () => {
    switch (verdict_label) {
      case 'Supported':
        return <CheckCircle2 size={20} color="#000000" />;
      case 'Refuted':
        return <XCircle size={20} color="#000000" />;
      default:
        return <HelpCircle size={20} color="#000000" />;
    }
  };

  return (
    <div className="summary-card">
      <div className="verdict-header">
        <div className="verdict-badge-large">
          {renderVerdictIcon()}
          <span>{verdict_label}</span>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <span className="confidence-pill">Confidence: {(confidence * 100).toFixed(1)}%</span>
          <span className="confidence-pill">Score: {overall_score.toFixed(1)} / 100</span>
        </div>
      </div>

      <p className="summary-text">{summary}</p>

      {is_uncertain && (
        <div className="uncertainty-box">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontWeight: 600, marginBottom: '0.25rem' }}>
            <AlertCircle size={16} />
            <span>Uncertainty Gate Active (Abstention Guard)</span>
          </div>
          <p>{uncertainty_reason}</p>
        </div>
      )}

      {factors && (
        <div className="score-grid">
          {Object.entries(factors).map(([key, val]) => (
            <div key={key} className="factor-card">
              <div className="factor-title">{key.replace('_', ' ')}</div>
              <div className="factor-score-num">{val.score.toFixed(1)}</div>
              <div className="factor-bar-bg">
                <div
                  className="factor-bar-fill"
                  style={{ width: `${Math.min(100, Math.max(0, val.score))}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
