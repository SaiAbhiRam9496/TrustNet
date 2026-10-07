import React from 'react';
import { ShieldCheck, Cpu } from 'lucide-react';

export default function Header({ isConnected }) {
  return (
    <header className="header-bar">
      <div className="brand">
        <ShieldCheck size={28} color="#ffffff" />
        <span className="brand-title">TrustNet</span>

      </div>
      <div className="status-pill">
        <span className="pulse-dot" />
        <Cpu size={16} />
        <span>{isConnected ? "Backend API Online" : "Connecting..."}</span>
      </div>
    </header>
  );
}
