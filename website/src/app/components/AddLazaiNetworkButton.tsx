"use client";

// Add type for window.ethereum
interface Ethereum {
  request: (args: { method: string; params?: any[] }) => Promise<any>;
}
declare global {
  interface Window {
    ethereum?: Ethereum;
  }
}

export default function AddLazaiNetworkButton() {
  const handleClick = () => {
    if (typeof window !== "undefined" && window.ethereum) {
      window.ethereum.request({
        method: "wallet_addEthereumChain",
        params: [
          {
            chainId: "0x20A56", // 133718 in hex
            chainName: "LazAI (Pre Testnet)",
            nativeCurrency: {
              name: "LAZAI",
              symbol: "LAZAI",
              decimals: 18,
            },
            rpcUrls: ["https://testnet.lazai.network"],
            blockExplorerUrls: [
              "https://testnet.lazai.network",
            ],
          },
        ],
      });
    } else {
      alert("MetaMask is not detected.");
    }
  };

  return (
    <button
      style={{
        background: "#1a73e8",
        color: "white",
        border: "none",
        borderRadius: "4px",
        padding: "10px 20px",
        fontSize: "16px",
        cursor: "pointer",
        marginTop: "16px",
      }}
      onClick={handleClick}
    >
      Add LazAI Network to MetaMask
    </button>
  );
}
