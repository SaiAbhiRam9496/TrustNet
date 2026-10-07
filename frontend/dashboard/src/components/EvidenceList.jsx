import React from 'react';
import { ExternalLink } from 'lucide-react';

export default function EvidenceList({ evidence }) {
  if (!evidence || evidence.length === 0) return null;

  return (
    <div className="inspector-section">
      <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '1rem', color: 'var(--text-primary)' }}>
        Retrieved & Verified Evidence Passages ({evidence.length})
      </h3>
      <div className="data-list">
        {evidence.map((item, idx) => (
          <div key={idx} className="data-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.4rem' }}>
              <div>
                <span className="evidence-title">{item.title || item.source_domain}</span>
                <a
                  href={item.url}
                  target="_blank"
                  rel="noreferrer"
                  style={{ display: 'inline-flex', alignItems: 'center', gap: '0.2rem', marginLeft: '0.5rem', color: 'var(--text-muted)' }}
                >
                  <ExternalLink size={12} />
                </a>
              </div>
              <div style={{ display: 'flex', gap: '0.4rem' }}>
                <span className="tag-mono">Tier: {item.domain_authority_tier}</span>
                <span className="stance-badge">{item.stance}</span>
              </div>
            </div>
            <p className="evidence-snippet">"{item.snippet}"</p>
          </div>
        ))}
      </div>
    </div>
  );
}
