"use client";

import { WagmiProvider, createConfig, http } from "wagmi";
import { defineChain } from "viem";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  RainbowKitProvider,
  connectorsForWallets,
} from "@rainbow-me/rainbowkit";
import {
  rainbowWallet,
  metaMaskWallet,
  coinbaseWallet,
  phantomWallet,
} from "@rainbow-me/rainbowkit/wallets";

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

// Configure specific wallets with RainbowKit
const connectors = connectorsForWallets(
  [
    {
      groupName: "Recommended",
      wallets: [rainbowWallet, metaMaskWallet, coinbaseWallet, phantomWallet],
    },
  ],
  {
    appName: "Alith",
    projectId: "your-project-id", // This won't actually be used since we're not including WalletConnect
  }
);

const config = createConfig({
  connectors,
  chains: [lazaiTestnet],
  transports: {
    [lazaiTestnet.id]: http(),
  },
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
