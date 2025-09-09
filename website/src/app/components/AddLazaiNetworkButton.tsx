"use client";

import { useChainId, useSwitchChain } from "wagmi";
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
          return (
            <WalletButtonInner
              account={account}
              chain={chain}
              openAccountModal={openAccountModal}
              openChainModal={openChainModal}
              openConnectModal={openConnectModal}
              authenticationStatus={authenticationStatus}
              rainbowMounted={rainbowMounted}
              mounted={mounted}
            />
          );
        }}
      </ConnectButton.Custom>
    </div>
  );
}

function WalletButtonInner({
  account,
  chain,
  openAccountModal,
  openChainModal,
  openConnectModal,
  authenticationStatus,
  rainbowMounted,
  mounted,
}: {
  account: any;
  chain: any;
  openAccountModal: () => void;
  openChainModal: () => void;
  openConnectModal: () => void;
  authenticationStatus: any;
  rainbowMounted: boolean;
  mounted: boolean;
}) {
  const chainId = useChainId();
  const { switchChain } = useSwitchChain();

  const isOnLazAI = chainId === lazaiTestnet.id;

  const handleAddNetwork = async () => {
    try {
      await switchChain({ chainId: lazaiTestnet.id });
    } catch (error) {
      console.error("Failed to switch to LazAI network:", error);
      if (error instanceof Error && error.message.includes("rejected")) {
        return;
      }
      alert(
        "Failed to switch to LazAI network. Please add it manually in your wallet."
      );
    }
  };

  const ready = rainbowMounted && mounted;
  const connected =
    ready &&
    account &&
    chain &&
    (!authenticationStatus || authenticationStatus === "authenticated");

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
    <div className={styles.connectedContainer}>
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
        <button
          className={styles.networkButton}
          onClick={handleAddNetwork}
        >
          Switch to LazAI Network
        </button>
      )}
      {isOnLazAI && (
        <span className={styles.successText}>✓ LazAI Testnet</span>
      )}
    </div>
  );
}

export default function AddLazaiNetworkButton() {
  return <WalletButton />;
}
