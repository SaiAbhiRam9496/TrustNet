import React from 'react';
import { Search, Loader2 } from 'lucide-react';

const PRESETS = [
  "In a letter to Steve Jobs, Sean Connery refused to appear in an apple commercial.",
  "NASA faked the moon landing in 1969.",
  "UNESCO declared Nadar community as the most ancient race in the world."
];

export default function ClaimInput({ claim, setClaim, onSubmit, isLoading }) {
  return (
    <div className="claim-card">
      <form onSubmit={onSubmit} className="claim-form">
        <textarea
          className="claim-textarea"
          value={claim}
          onChange={(e) => setClaim(e.target.value)}
          placeholder="Enter any claim or statement to verify with evidence..."
          disabled={isLoading}
        />
        <div className="form-actions">
          <div className="preset-pills">
            {PRESETS.map((preset, idx) => (
              <button
                key={idx}
                type="button"
                className="preset-btn"
                onClick={() => setClaim(preset)}
                disabled={isLoading}
              >
                Sample #{idx + 1}
              </button>
            ))}
          </div>
          <button
            type="submit"
            className="submit-btn"
            disabled={isLoading || !claim.trim()}
          >
            {isLoading ? (
              <>
                <Loader2 size={16} className="spin" />
                <span>Analyzing Pipeline...</span>
              </>
            ) : (
              <>
                <Search size={16} />
                <span>Verify Claim</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
