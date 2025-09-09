"use client";

import { useSwitchChain } from "wagmi";
import { ConnectButton } from "@rainbow-me/rainbowkit";
import { useEffect, useState } from "react";
import { defineChain } from "viem";
import styles from "./AddLazaiNetworkButton.module.css";
import "@rainbow-me/rainbowkit/styles.css";

// LazAI Testnet configuration
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
      name: "LazAI Explorer",
      url: "https://testnet-explorer.lazai.network",
    },
  },
});

function WalletButton() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div className={styles.container}>
        <button className={styles.connectButton} disabled>
          Loading...
        </button>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <ConnectButton.Custom>
        {({
          account,
          chain,
          openAccountModal,
          openChainModal,
          openConnectModal,
          authenticationStatus,
          mounted: rainbowMounted,
        }) => {
          const ready = rainbowMounted && mounted;
          const connected =
            ready &&
            account &&
            chain &&
            (!authenticationStatus || authenticationStatus === "authenticated");

          // Use Wagmi hooks here within the RainbowKit context
          const { switchChain } = useSwitchChain();
          const isOnLazAI = chain?.id === lazaiTestnet.id;

          const handleAddNetwork = async () => {
            if (connected) {
              try {
                await switchChain({ chainId: lazaiTestnet.id });
              } catch (error) {
                console.error("Failed to switch to LazAI network:", error);
                if (
                  error instanceof Error &&
                  error.message.includes("rejected")
                ) {
                  return;
                }
                alert(
                  "Failed to switch to LazAI network. Please add it manually in your wallet."
                );
              }
            } else {
              alert("Please connect your wallet first.");
            }
          };

          if (!ready) {
            return (
              <button className={styles.connectButton} disabled>
                Loading...
              </button>
            );
          }

          if (!connected) {
            return (
              <button
                className={styles.connectButton}
                onClick={openConnectModal}
              >
                Connect Wallet
              </button>
            );
          }

          return (
            <div className={styles.walletContainer}>
              <div className={styles.walletInfo}>
                <button
                  className={styles.accountButton}
                  onClick={openAccountModal}
                >
                  {account.displayName}
                  {account.displayBalance ? ` (${account.displayBalance})` : ""}
                </button>
                <button className={styles.chainButton} onClick={openChainModal}>
                  {chain.hasIcon && (
                    <div className={styles.chainIcon}>
                      {chain.iconUrl && (
                        <img
                          alt={chain.name ?? "Chain icon"}
                          src={chain.iconUrl}
                          className={styles.chainIconImage}
                        />
                      )}
                    </div>
                  )}
                  {chain.name}
                </button>
              </div>
              {!isOnLazAI && (
                <button className={styles.button} onClick={handleAddNetwork}>
                  Add LazAI Network
                </button>
              )}
              {isOnLazAI && (
                <span className={styles.successText}>
                  ✓ Connected to LazAI Testnet
                </span>
              )}
            </div>
          );
        }}
      </ConnectButton.Custom>
    </div>
  );
}

export default function AddLazaiNetworkButton() {
  return <WalletButton />;
}
