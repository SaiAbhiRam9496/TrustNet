import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import ClaimInput from './components/ClaimInput';
import SummaryCard from './components/SummaryCard';
import PipelineInspector from './components/PipelineInspector';
import EvidenceList from './components/EvidenceList';

const API_BASE = "http://127.0.0.1:8000/api/v1";

export default function App() {
  const [claim, setClaim] = useState('');
  const [result, setResult] = useState(null);
  const [liveInspector, setLiveInspector] = useState(null);
  const [activeStage, setActiveStage] = useState('stage1');
  const [stageStatus, setStageStatus] = useState({
    stage1: 'pending',
    stage2: 'pending',
    stage3: 'pending',
    stage4: 'pending',
    stage5: 'pending',
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isConnected, setIsConnected] = useState(false);

  // Check health on mount
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((res) => res.ok ? setIsConnected(true) : setIsConnected(false))
      .catch(() => setIsConnected(false));
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!claim.trim()) return;

    setIsLoading(true);
    setError(null);
    setResult(null);
    setLiveInspector({});
    setActiveStage('stage1');
    setStageStatus({
      stage1: 'running',
      stage2: 'pending',
      stage3: 'pending',
      stage4: 'pending',
      stage5: 'pending',
    });

    try {
      const response = await fetch(`${API_BASE}/analyze/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ claim }),
      });

      if (!response.ok) {
        throw new Error(`Stream returned status ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';

        for (const part of parts) {
          const trimmed = part.trim();
          if (!trimmed.startsWith('data:')) continue;

          const jsonStr = trimmed.replace(/^data:\s*/, '');
          if (!jsonStr) continue;

          try {
            const payload = JSON.parse(jsonStr);

            if (payload.status === 'error') {
              throw new Error(payload.error || 'Pipeline execution error.');
            }

            if (payload.pipeline_inspector) {
              setLiveInspector(payload.pipeline_inspector);
            }

            if (payload.stage) {
              const current = payload.stage;
              setActiveStage(current);

              setStageStatus((prev) => {
                const next = { ...prev };
                if (current === 'stage1') {
                  next.stage1 = 'done';
                  next.stage2 = 'running';
                } else if (current === 'stage2') {
                  next.stage2 = 'done';
                  next.stage3 = 'running';
                } else if (current === 'stage3') {
                  next.stage3 = 'done';
                  next.stage4 = 'running';
                } else if (current === 'stage4') {
                  next.stage4 = 'done';
                  next.stage5 = 'running';
                } else if (current === 'stage5') {
                  next.stage5 = 'done';
                }
                return next;
              });
            }

            if (payload.status === 'complete' && payload.result) {
              setResult(payload.result);
            }
          } catch (parseErr) {
            console.warn('Error parsing SSE chunk:', parseErr);
          }
        }
      }
    } catch (streamErr) {
      console.warn('SSE stream failed, falling back to standard /analyze endpoint:', streamErr);
      try {
        const fallbackRes = await fetch(`${API_BASE}/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ claim }),
        });
        if (!fallbackRes.ok) {
          const errData = await fallbackRes.json().catch(() => ({}));
          throw new Error(errData.message || `Server returned status ${fallbackRes.status}`);
        }
        const data = await fallbackRes.json();
        setResult(data);
        if (data.pipeline_inspector) {
          setLiveInspector(data.pipeline_inspector);
        }
      } catch (fallbackErr) {
        setError(fallbackErr.message || 'Failed to connect to TrustNet backend API.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Header isConnected={isConnected} />

      <main>
        <ClaimInput
          claim={claim}
          setClaim={setClaim}
          onSubmit={handleSubmit}
          isLoading={isLoading}
        />

        {error && (
          <div className="uncertainty-box" style={{ borderColor: '#ef4444', marginBottom: '2rem' }}>
            <strong>Analysis Error:</strong> {error}
          </div>
        )}

        {(isLoading || liveInspector) && (
          <PipelineInspector
            data={liveInspector}
            activeStage={activeStage}
            setActiveStage={setActiveStage}
            stageStatus={stageStatus}
          />
        )}

        {result && (
          <>
            <SummaryCard data={result} />
            <EvidenceList evidence={result.evidence} />
          </>
        )}
      </main>


    </div>
  );
}

