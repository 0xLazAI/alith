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
  walletConnectWallet,
} from "@rainbow-me/rainbowkit/wallets";

import { useEffect, useState } from "react";
import "@rainbow-me/rainbowkit/styles.css";

// Suppress React DOM warnings for QR code components
if (typeof window !== "undefined") {
  // Store original console methods
  const originalError = console.error;
  const originalWarn = console.warn;

  // Override console.error
  console.error = (...args: any[]) => {
    const message = String(args[0] || "");
    const shouldSuppress =
      message.includes("errorCorrection") ||
      message.includes("React does not recognize") ||
      (message.includes("DOM element") && message.includes("prop")) ||
      message.includes("custom attribute");

    if (shouldSuppress) {
      return; // Suppress the warning
    }
    originalError.apply(console, args);
  };

  // Override console.warn for good measure
  console.warn = (...args: any[]) => {
    const message = String(args[0] || "");
    const shouldSuppress =
      message.includes("errorCorrection") ||
      message.includes("React does not recognize");

    if (shouldSuppress) {
      return; // Suppress the warning
    }
    originalWarn.apply(console, args);
  };
}

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

// Configure wallets including WalletConnect with your project ID
const connectors = connectorsForWallets(
  [
    {
      groupName: "Recommended",
      wallets: [
        rainbowWallet,
        metaMaskWallet,
        coinbaseWallet,
        phantomWallet,
        walletConnectWallet,
      ],
    },
  ],
  {
    appName: "Alith",
    projectId: "e5d0033a3ef41588ef730193c29f5391", // Your WalletConnect project ID
  }
);

const config = createConfig({
  connectors,
  chains: [lazaiTestnet],
  transports: {
    [lazaiTestnet.id]: http(),
  },
  ssr: false,
  multiInjectedProviderDiscovery: false,
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
          appInfo={{
            appName: "Alith",
            learnMoreUrl: "https://alith.ai",
          }}
        >
          {children}
        </RainbowKitProvider>
      </QueryClientProvider>
    </WagmiProvider>
  );
}
