"use client";

export default function LazaiCards() {
  return (
    <div style={{ display: 'flex', gap: '32px', marginTop: '32px', flexWrap: 'wrap', justifyContent: 'center' }}>
      <a
        href="#building-on-lazai"
        style={{
          flex: '1 1 300px',
          background: 'linear-gradient(135deg, #e0e7ff 0%, #f0fdfa 100%)',
          borderRadius: '16px',
          padding: '32px',
          minWidth: '280px',
          boxShadow: '0 4px 24px rgba(80, 112, 255, 0.10)',
          textDecoration: 'none',
          color: '#1a237e',
          transition: 'transform 0.15s, box-shadow 0.15s',
          cursor: 'pointer',
          display: 'block',
          fontWeight: 500,
          fontSize: '1.1rem',
          textAlign: 'center',
          outline: 'none',
          border: '2px solid transparent',
        }}
        onMouseOver={e => {
          e.currentTarget.style.transform = 'translateY(-4px) scale(1.03)';
          e.currentTarget.style.boxShadow = '0 8px 32px rgba(80,112,255,0.18)';
        }}
        onMouseOut={e => {
          e.currentTarget.style.transform = 'none';
          e.currentTarget.style.boxShadow = '0 4px 24px rgba(80,112,255,0.10)';
        }}
      >
        <h3 style={{ marginTop: 0, marginBottom: '12px', fontSize: '1.3rem' }}>🚀 Building on LazAI</h3>
        <p style={{ margin: 0 }}>
        </p>
      </a>
      <a
        href="#data-provider"
        style={{
          flex: '1 1 300px',
          background: 'linear-gradient(135deg, #fffbe7 0%, #e0f7fa 100%)',
          borderRadius: '16px',
          padding: '32px',
          minWidth: '280px',
          boxShadow: '0 4px 24px rgba(255, 193, 7, 0.10)',
          textDecoration: 'none',
          color: '#7b3f00',
          transition: 'transform 0.15s, box-shadow 0.15s',
          cursor: 'pointer',
          display: 'block',
          fontWeight: 500,
          fontSize: '1.1rem',
          textAlign: 'center',
          outline: 'none',
          border: '2px solid transparent',
        }}
        onMouseOver={e => {
          e.currentTarget.style.transform = 'translateY(-4px) scale(1.03)';
          e.currentTarget.style.boxShadow = '0 8px 32px rgba(255,193,7,0.18)';
        }}
        onMouseOut={e => {
          e.currentTarget.style.transform = 'none';
          e.currentTarget.style.boxShadow = '0 4px 24px rgba(255,193,7,0.10)';
        }}
      >
        <h3 style={{ marginTop: 0, marginBottom: '12px', fontSize: '1.3rem' }}>📊 Data Provider</h3>
        <p style={{ margin: 0 }}>
          Contribute valuable data, computation, or resources to the LazAI network. Help power decentralized AI, earn rewards, and support the growth of a transparent, privacy-preserving AI ecosystem.
        </p>
      </a>
    </div>
  );
} 