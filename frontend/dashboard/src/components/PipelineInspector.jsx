import React from 'react';
import { Layers, Search, Database, Scale, Calculator, Loader2, CheckCircle2, Clock } from 'lucide-react';

export default function PipelineInspector({ data, activeStage, setActiveStage, stageStatus = {} }) {
  if (!data) return null;

  const inspector = data.pipeline_inspector || data;
  const stage1 = inspector.stage1_decomposition || {};
  const stage2 = inspector.stage2_retrieval || {};
  const stage3 = inspector.stage3_ranking || {};
  const stage4 = inspector.stage4_nli || {};
  const stage5 = inspector.stage5_scoring || {};

  const TABS = [
    { id: 'stage1', label: '1. Decomposition', icon: Layers, dataKey: 'stage1_decomposition' },
    { id: 'stage2', label: '2. Raw Retrieval', icon: Search, dataKey: 'stage2_retrieval' },
    { id: 'stage3', label: '3. Hybrid Ranking', icon: Database, dataKey: 'stage3_ranking' },
    { id: 'stage4', label: '4. DeBERTa NLI', icon: Scale, dataKey: 'stage4_nli' },
    { id: 'stage5', label: '5. Scoring Math', icon: Calculator, dataKey: 'stage5_scoring' },
  ];

  const renderStatusIcon = (stageId, dataKey) => {
    const status = stageStatus[stageId] || (inspector[dataKey] ? 'done' : 'pending');
    if (status === 'done' || inspector[dataKey]) {
      return <CheckCircle2 size={12} style={{ color: '#ffffff' }} />;
    } else if (status === 'running') {
      return <Loader2 size={12} className="spin" style={{ color: '#ffffff' }} />;
    } else {
      return <Clock size={12} style={{ color: 'var(--text-muted)' }} />;
    }
  };

  return (
    <div className="inspector-section">
      <div className="inspector-header">
        <div className="inspector-title-row">
          <span className="inspector-title">Live Pipeline Inspector (Stage-by-Stage Transparency)</span>
          <span className="tag-mono">Faculty Demo Mode</span>
        </div>
        <div className="tabs-row">
          {TABS.map((t) => {
            const Icon = t.icon;
            return (
              <button
                key={t.id}
                className={`tab-btn ${activeStage === t.id ? 'active' : ''}`}
                onClick={() => setActiveStage && setActiveStage(t.id)}
              >
                {renderStatusIcon(t.id, t.dataKey)}
                <Icon size={14} />
                <span>{t.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="stage-panel">
        {activeStage === 'stage1' && (
          inspector.stage1_decomposition ? (
            <div className="data-list">
              <h4 style={{ color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>Atomic Subclaims Generated (Gemini)</h4>
              {(stage1.subclaims || []).map((sc, idx) => (
                <div key={idx} className="data-card">
                  <p className="subclaim-text">Subclaim [{idx + 1}]: {sc}</p>
                </div>
              ))}
              <h4 style={{ color: 'var(--text-secondary)', marginTop: '1rem', marginBottom: '0.5rem' }}>Targeted Search Queries</h4>
              <div className="tags-row">
                {(stage1.search_queries || []).map((q, idx) => (
                  <span key={idx} className="tag-mono">"{q}"</span>
                ))}
              </div>
            </div>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', padding: '1rem 0' }}>
              <Loader2 size={16} className="spin" />
              <span>Stage 1: Decomposing claim into subclaims & search queries...</span>
            </div>
          )
        )}

        {activeStage === 'stage2' && (
          inspector.stage2_retrieval ? (
            <div className="data-list">
              <p style={{ color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                Total Raw Web Sources Retrieved: <strong>{stage2.total_raw_sources || 0}</strong>
              </p>
              {(stage2.urls || []).map((url, idx) => (
                <div key={idx} className="data-card">
                  <span className="evidence-url">[{idx + 1}] {url}</span>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', padding: '1rem 0' }}>
              <Loader2 size={16} className="spin" />
              <span>Stage 2: Scraping web sources via Tavily search...</span>
            </div>
          )
        )}

        {activeStage === 'stage3' && (
          inspector.stage3_ranking ? (
            <div className="data-list">
              <p style={{ color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                BM25 + SentenceTransformer RRF & CrossEncoder Top Passages
              </p>
              {(stage3.top_passages || []).map((item, idx) => (
                <div key={idx} className="data-card">
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                    <span className="evidence-title">[{idx + 1}] {item.source_domain}</span>
                    <span className="tag-mono">Score: {item.relevance_score?.toFixed(4)}</span>
                  </div>
                  <p className="evidence-url">{item.url}</p>
                  <p className="evidence-snippet">"{item.snippet_preview}..."</p>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', padding: '1rem 0' }}>
              <Loader2 size={16} className="spin" />
              <span>Stage 3: Hybrid BM25+Dense ranking and Cross-Encoder reranking...</span>
            </div>
          )
        )}

        {activeStage === 'stage4' && (
          inspector.stage4_nli ? (
            <div className="data-list">
              <h4 style={{ color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>DeBERTa-v3 Stance Classification</h4>
              {(stage4.subclaim_stances || []).map((item, idx) => (
                <div key={idx} className="data-card">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <p className="subclaim-text" style={{ margin: 0 }}>"{item.subclaim}"</p>
                    <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                      <span className="stance-badge">{item.stance}</span>
                      <span className="tag-mono">Conf: {(item.nli_conf * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', padding: '1rem 0' }}>
              <Loader2 size={16} className="spin" />
              <span>Stage 4: Running DeBERTa-v3 NLI stance classification...</span>
            </div>
          )
        )}

        {activeStage === 'stage5' && (
          inspector.stage5_scoring ? (
            <div className="data-list">
              <div className="data-card">
                <h4 style={{ marginBottom: '0.75rem' }}>Empirical Dynamic Confidence Formula</h4>
                <div className="code-box">
                  Confidence = 0.45 * |AgreementRatio| + 0.30 * DomainAuthorityScore + 0.25 * AvgNLISoftmax{"\n\n"}
                  Inputs for this claim:{"\n"}
                  - AgreementRatio        = {stage5.agreement_ratio?.toFixed(4)}{"\n"}
                  - DomainAuthorityScore  = {stage5.domain_authority_score?.toFixed(4)}{"\n"}
                  - AvgNLISoftmax         = {stage5.avg_nli_softmax?.toFixed(4)}{"\n\n"}
                  Calculated Dynamic Confidence = {stage5.dynamic_confidence?.toFixed(4)} ({(stage5.dynamic_confidence * 100).toFixed(1)}%)
                </div>
              </div>
            </div>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', padding: '1rem 0' }}>
              <Loader2 size={16} className="spin" />
              <span>Stage 5: Computing dynamic confidence math & verdict synthesis...</span>
            </div>
          )
        )}
      </div>
    </div>
  );
}

