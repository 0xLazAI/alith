"use client";

import { WagmiProvider, createConfig, http } from "wagmi";
import { defineChain } from "viem";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RainbowKitProvider, getDefaultConfig } from "@rainbow-me/rainbowkit";

import { useEffect, useState } from "react";
import "@rainbow-me/rainbowkit/styles.css";

const lazaiTestnet = defineChain({
  id: 133718,
  name: "LazAI Testnet",
  network: "lazai-testnet",
  nativeCurrency: {
    decimals: 18,
    name: "LAZAI",
    symbol: "LAZAI",
  },
  rpcUrls: {
    public: { http: ["https://testnet.lazai.network"] },
    default: { http: ["https://testnet.lazai.network"] },
  },
  blockExplorers: {
    default: {
      name: "LazAI Testnet Explorer",
      url: "https://testnet-explorer.lazai.network",
    },
  },
});

// Use getDefaultConfig but with custom projectId to minimize WalletConnect issues
const config = getDefaultConfig({
  appName: "Alith",
  projectId: "alith-local-dev",
  chains: [lazaiTestnet],
  ssr: false,
});

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
      refetchOnWindowFocus: false,
    },
  },
});

export function Providers({ children }: { children: React.ReactNode }) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return <div>{children}</div>;
  }

  return (
    <WagmiProvider config={config}>
      <QueryClientProvider client={queryClient}>
        <RainbowKitProvider
          modalSize="compact"
          initialChain={lazaiTestnet}
          showRecentTransactions={true}
        >
          {children}
        </RainbowKitProvider>
      </QueryClientProvider>
    </WagmiProvider>
  );
}
