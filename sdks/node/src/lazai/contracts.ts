// Local Devnet Contract addresses

export const DEFAULT_DATA_REGISTRY_CONTRACT_ADDRESS =
  "0xEAd077726dC83ecF385e3763ed4A0A50E8Ac5AA0";
export const DEFAULT_VERIFIED_COMPUTING_CONTRACT_ADDRESS =
  "0x815da22D880E3560bCEcc85b6e4938b30c8202C4";
export const DEFAULT_DATA_ANCHORING_TOKEN_CONTRACT_ADDRESS =
  "0x2eD344c586303C98FC3c6D5B42C5616ED42f9D9d";
export const DEFAULT_QUERY_CONTRACT_ADDRESS =
  "0xE747fd70269a8a540403ddE802D6906CB18C9F50";
export const DEFAULT_INFERENCE_CONTRACT_ADDRESS =
  "0xbb969eaafB3A7124b8dCdf9a6d5Cd5BAa0381361";
export const DEFAULT_TRAINING_CONTRACT_ADDRESS =
  "0xb578AB78bb4780D9007Cc836b358468467814B3E";
export const DEFAULT_SETTLEMENT_CONTRACT_ADDRESS =
  "0xBE94646A0C6C1032c289Eea47169798e09dB5299";

// Testnet Contract Addresses

export const TESTNET_ADMIN_ADDRESS =
  "0x34d9E02F9bB4E4C8836e38DF4320D4a79106F194";
export const TESTNET_DATA_REGISTRY_CONTRACT_ADDRESS =
  "0xE7753EeBCA82849D6b19E6689B350f87318A8998";
export const TESTNET_VERIFIED_COMPUTING_CONTRACT_ADDRESS =
  "0x87E43F24Efc2284fd2BBF11CC80d6fcF3E0AD474";
export const TESTNET_DATA_ANCHORING_TOKEN_CONTRACT_ADDRESS =
  "0xD59CDFFEb65aCc539994e41D0B40efF61bE37118";
export const TESTNET_QUERY_CONTRACT_ADDRESS =
  "0x5D7fC5A04328b95cae017B664A8e95fa25Ca3e98";
export const TESTNET_INFERENCE_CONTRACT_ADDRESS =
  "0x69BD47252510573995b22ae227560E879b193738";
export const TESTNET_TRAINING_CONTRACT_ADDRESS =
  "0x7e6646feEC69df501D942e16CBE2d14B7bBEC853";
export const TESTNET_SETTLEMENT_CONTRACT_ADDRESS =
  "0xF1398c4Bb36245750393A2511dA8bF1F7828F979";

// Contract ABIs

export const DATA_REGISTRY_CONTRACT_ABI = [
  {
    constant: true,
    inputs: [],
    name: "name",
    outputs: [{ name: "", type: "string" }],
    payable: false,
    stateMutability: "view",
    type: "function",
  },
  {
    constant: true,
    inputs: [],
    name: "version",
    outputs: [{ name: "", type: "uint256" }],
    payable: false,
    stateMutability: "pure",
    type: "function",
  },
  {
    constant: true,
    inputs: [],
    name: "token",
    outputs: [{ name: "", type: "address" }],
    payable: false,
    stateMutability: "view",
    type: "function",
  },
  {
    constant: true,
    inputs: [],
    name: "verifiedComputing",
    outputs: [{ name: "", type: "address" }],
    payable: false,
    stateMutability: "view",
    type: "function",
  },
  {
    inputs: [{ name: "newVerifiedComputing", type: "address" }],
    name: "updateVerifiedComputing",
    outputs: [],
    stateMutability: "nonpayable",
    type: "function",
  },
  {
    constant: true,
    inputs: [],
    name: "publicKey",
    outputs: [{ name: "", type: "string" }],
    payable: false,
    stateMutability: "view",
    type: "function",
  },
  {
    inputs: [{ name: "newPublicKey", type: "string" }],
    name: "updatePublicKey",
    outputs: [],
    stateMutability: "nonpayable",
    type: "function",
  },
  {
    inputs: [
      { name: "url", type: "string" },
      { name: "hash", type: "string" },
    ],
    name: "addFile",
    outputs: [{ name: "", type: "uint256" }],
    stateMutability: "nonpayable",
    type: "function",
  },
  {
    inputs: [
      { name: "url", type: "string" },
      { name: "hash", type: "string" },
      { name: "ownerAddress", type: "address" },
      {
        components: [
          { name: "account", type: "address" },
          { name: "key", type: "string" },
        ],
        name: "permissions",
        type: "tuple[]",
      },
    ],
    name: "addFileWithPermissions",
    outputs: [{ name: "", type: "uint256" }],
    stateMutability: "nonpayable",
    type: "function",
  },
  {
    inputs: [
      { name: "fileId", type: "uint256" },
      { name: "account", type: "address" },
      { name: "key", type: "string" },
    ],
    name: "addPermissionForFile",
    outputs: [],
    stateMutability: "nonpayable",
    type: "function",
  },
  {
    constant: true,
    inputs: [{ name: "fileId", type: "uint256" }],
    name: "getFile",
    outputs: [
      {
        components: [
          { name: "id", type: "uint256" },
          { name: "ownerAddress", type: "address" },
          { name: "url", type: "string" },
          { name: "hash", type: "string" },
          { name: "proofIndex", type: "uint256" },
          { name: "rewardAmount", type: "uint256" },
        ],
        name: "",
        type: "tuple",
      },
    ],
    stateMutability: "view",
    type: "function",
  },
  {
    constant: true,
    inputs: [{ name: "url", type: "string" }],
    name: "getFileIdByUrl",
    outputs: [{ name: "", type: "uint256" }],
    stateMutability: "view",
    type: "function",
  },
  {
    constant: true,
    inputs: [
      { name: "fileId", type: "uint256" },
      { name: "account", type: "address" },
    ],
    name: "getFilePermission",
    outputs: [{ name: "", type: "string" }],
    stateMutability: "view",
    type: "function",
  },
  {
    constant: true,
    inputs: [
      { name: "fileId", type: "uint256" },
      { name: "index", type: "uint256" },
    ],
    name: "getFileProof",
    outputs: [
      {
        components: [
          { name: "timestamp", type: "uint256" },
          { name: "hash", type: "bytes32" },
          { name: "signature", type: "bytes" },
        ],
        name: "",
        type: "tuple",
      },
    ],
    stateMutability: "view",
    type: "function",
  },
  {
    constant: true,
    inputs: [],
    name: "filesCount",
    outputs: [{ name: "", type: "uint256" }],
    stateMutability: "view",
    type: "function",
  },
  {
    inputs: [
      { internalType: "uint256", name: "fileId", type: "uint256" },
      {
        components: [
          { internalType: "bytes", name: "signature", type: "bytes" },
          {
            components: [
              {
                internalType: "uint256",
                name: "id",
                type: "uint256",
              },
              {
                internalType: "uint256",
                name: "score",
                type: "uint256",
              },
              {
                internalType: "string",
                name: "fileUrl",
                type: "string",
              },
              {
                internalType: "string",
                name: "proofUrl",
                type: "string",
              },
            ],
            internalType: "struct ProofData",
            name: "data",
            type: "tuple",
          },
        ],
        internalType: "struct Proof",
        name: "proof",
        type: "tuple",
      },
    ],
    name: "addProof",
    outputs: [],
    stateMutability: "nonpayable",
    type: "function",
  },
  {
    inputs: [
      { name: "fileId", type: "uint256" },
      { name: "proofIndex", type: "uint256" },
    ],
    name: "requestReward",
    outputs: [],
    stateMutability: "nonpayable",
    type: "function",
  },
];
export const VERIFIED_COMPUTING_CONTRACT_ABI = [
  {
    type: "enum",
    name: "NodeStatus",
    inputs: [
      { name: "None", type: "uint8" },
      { name: "Active", type: "uint8" },
      { name: "Removed", type: "uint8" },
    ],
  },
  {
    type: "enum",
    name: "JobStatus",
    inputs: [
      { name: "None", type: "uint8" },
      { name: "Submitted", type: "uint8" },
      { name: "Completed", type: "uint8" },
      { name: "Canceled", type: "uint8" },
    ],
  },
  {
    type: "struct",
    name: "Job",
    components: [
      { name: "fileId", type: "uint256" },
      { name: "bidAmount", type: "uint256" },
      { name: "status", type: "uint8" },
      { name: "addedTimestamp", type: "uint256" },
      { name: "ownerAddress", type: "address" },
      { name: "nodeAddress", type: "address" },
    ],
  },
  {
    type: "struct",
    name: "NodeInfo",
    components: [
      { name: "nodeAddress", type: "address" },
      { name: "url", type: "string" },
      { name: "status", type: "uint8" },
      { name: "amount", type: "uint256" },
      { name: "jobsCount", type: "uint256" },
      { name: "publicKey", type: "string" },
    ],
  },
  {
    constant: true,
    inputs: [],
    name: "version",
    outputs: [{ name: "", type: "uint256" }],
    stateMutability: "pure",
    type: "function",
  },
  {
    inputs: [],
    name: "pause",
    outputs: [],
    stateMutability: "nonpayable",
    type: "function",
  },
  {
    inputs: [],
    name: "unpause",
    outputs: [],
    stateMutability: "nonpayable",
    type: "function",
  },
  {
    constant: true,
    inputs: [],
    name: "nodeFee",
    outputs: [{ name: "", type: "uint256" }],
    stateMutability: "view",
    type: "function",
  },
  {
    inputs: [{ name: "newNodeFee", type: "uint256" }],
    name: "updateNodeFee",
    outputs: [],
    stateMutability: "nonpayable",
    type: "function",
  },
  {
    constant: true,
    inputs: [],
    name: "nodeList",
    outputs: [{ name: "", type: "address[]" }],
    stateMutability: "view",
    type: "function",
  },
  {
    constant: true,
    inputs: [{ name: "index", type: "uint256" }],
    name: "nodeListAt",
    outputs: [
      {
        components: [
          { name: "nodeAddress", type: "address" },
          { name: "url", type: "string" },
          { name: "status", type: "uint8" },
          { name: "amount", type: "uint256" },
          { name: "jobsCount", type: "uint256" },
          { name: "publicKey", type: "string" },
        ],
        type: "tuple",
      },
    ],
    stateMutability: "view",
    type: "function",
  },
  {
    constant: true,
    inputs: [],
    name: "nodesCount",
    outputs: [{ name: "", type: "uint256" }],
    stateMutability: "view",
    type: "function",
  },
  {
    constant: true,
    inputs: [],
    name: "activeNodesCount",
    outputs: [{ name: "", type: "uint256" }],
    stateMutability: "view",
    type: "function",
  },
  {
    constant: true,
    inputs: [],
    name: "activeNodeList",
    outputs: [{ name: "", type: "address[]" }],
    stateMutability: "view",
    type: "function",
  },
  {
    constant: true,
    inputs: [{ name: "index", type: "uint256" }],
    name: "activeNodeListAt",
    outputs: [
      {
        components: [
          { name: "nodeAddress", type: "address" },
          { name: "url", type: "string" },
          { name: "status", type: "uint8" },
          { name: "amount", type: "uint256" },
          { name: "jobsCount", type: "uint256" },
          { name: "publicKey", type: "string" },
        ],
        type: "tuple",
      },
    ],
    stateMutability: "view",
    type: "function",
  },
  {
    constant: true,
    inputs: [{ name: "nodeAddress", type: "address" }],
    name: "getNode",
    outputs: [
      {
        components: [
          { name: "nodeAddress", type: "address" },
          { name: "url", type: "string" },
          { name: "status", type: "uint8" },
          { name: "amount", type: "uint256" },
          { name: "jobsCount", type: "uint256" },
          { name: "publicKey", type: "string" },
        ],
        type: "tuple",
      },
    ],
    stateMutability: "view",
    type: "function",
  },
  {
    inputs: [
      { name: "nodeAddress", type: "address" },
      { name: "url", type: "string" },
      { name: "publicKey", type: "string" },
    ],
    name: "addNode",
    outputs: [],
    stateMutability: "nonpayable",
    type: "function",
  },
  {
    inputs: [{ name: "nodeAddress", type: "address" }],
    name: "removeNode",
    outputs: [],
    stateMutability: "nonpayable",
    type: "function",
  },
  {
    constant: true,
    inputs: [{ name: "nodeAddress", type: "address" }],
    name: "isNode",
    outputs: [{ name: "", type: "bool" }],
    stateMutability: "view",
    type: "function",
  },
  {
    inputs: [{ name: "fileId", type: "uint256" }],
    name: "requestProof",
    outputs: [],
    stateMutability: "payable",
    type: "function",
  },
  {
    inputs: [{ name: "jobId", type: "uint256" }],
    name: "completeJob",
    outputs: [],
    stateMutability: "nonpayable",
    type: "function",
  },
  {
    inputs: [{ name: "jobId", type: "uint256" }],
    name: "addProof",
    outputs: [],
    stateMutability: "nonpayable",
    type: "function",
  },
  {
    inputs: [{ name: "fileId", type: "uint256" }],
    name: "submitJob",
    outputs: [],
    stateMutability: "payable",
    type: "function",
  },
  {
    constant: true,
    inputs: [{ name: "fileId", type: "uint256" }],
    name: "fileJobIds",
    outputs: [{ name: "", type: "uint256[]" }],
    stateMutability: "view",
    type: "function",
  },
  {
    constant: true,
    inputs: [],
    name: "jobsCount",
    outputs: [{ name: "", type: "uint256" }],
    stateMutability: "view",
    type: "function",
  },
  {
    constant: true,
    inputs: [{ name: "jobId", type: "uint256" }],
    name: "getJob",
    outputs: [
      {
        components: [
          { name: "fileId", type: "uint256" },
          { name: "bidAmount", type: "uint256" },
          { name: "status", type: "uint8" },
          { name: "addedTimestamp", type: "uint256" },
          { name: "ownerAddress", type: "address" },
          { name: "nodeAddress", type: "address" },
        ],
        type: "tuple",
      },
    ],
    stateMutability: "view",
    type: "function",
  },
  {
    inputs: [],
    name: "claim",
    outputs: [],
    stateMutability: "nonpayable",
    type: "function",
  },
];
export const DATA_ANCHORING_TOKEN_CONTRACT = [
  {
    anonymous: false,
    inputs: [
      { indexed: true, internalType: "address", name: "to", type: "address" },
      {
        indexed: true,
        internalType: "uint256",
        name: "tokenId",
        type: "uint256",
      },
      {
        indexed: false,
        internalType: "string",
        name: "tokenURI",
        type: "string",
      },
    ],
    name: "TokenMinted",
    type: "event",
  },
  {
    inputs: [
      { internalType: "address", name: "to", type: "address" },
      { internalType: "uint256", name: "amount", type: "uint256" },
      { internalType: "string", name: "tokenURI_", type: "string" },
      { internalType: "bool", name: "verified_", type: "bool" },
    ],
    name: "mint",
    outputs: [],
    stateMutability: "nonpayable",
    type: "function",
  },
  {
    inputs: [{ internalType: "uint256", name: "tokenId", type: "uint256" }],
    name: "uri",
    outputs: [{ internalType: "string", name: "", type: "string" }],
    stateMutability: "view",
    type: "function",
  },
  {
    inputs: [{ internalType: "uint256", name: "tokenId", type: "uint256" }],
    name: "verified",
    outputs: [{ internalType: "bool", name: "", type: "bool" }],
    stateMutability: "view",
    type: "function",
  },
  {
    inputs: [
      { internalType: "uint256", name: "tokenId", type: "uint256" },
      { internalType: "bool", name: "verified_", type: "bool" },
    ],
    name: "setTokenVerified",
    outputs: [],
    stateMutability: "nonpayable",
    type: "function",
  },
  {
    inputs: [
      { internalType: "address", name: "account", type: "address" },
      { internalType: "uint256", name: "id", type: "uint256" },
    ],
    name: "balanceOf",
    outputs: [{ internalType: "uint256", name: "", type: "uint256" }],
    stateMutability: "view",
    type: "function",
  },
  {
    inputs: [
      { internalType: "address", name: "to", type: "address" },
      { internalType: "uint256[]", name: "ids", type: "uint256[]" },
      { internalType: "uint256[]", name: "amounts", type: "uint256[]" },
      { internalType: "string[]", name: "tokenURIs", type: "string[]" },
    ],
    name: "batchMint",
    outputs: [],
    stateMutability: "nonpayable",
    type: "function",
  },
];
export const AI_PROCESS_CONTRACT_ABI = [
  {
    type: "function",
    name: "version",
    inputs: [],
    outputs: [{ type: "uint256" }],
    stateMutability: "view",
  },
  {
    type: "function",
    name: "pause",
    inputs: [],
    outputs: [],
    stateMutability: "nonpayable",
  },
  {
    type: "function",
    name: "unpause",
    inputs: [],
    outputs: [],
    stateMutability: "nonpayable",
  },
  {
    type: "function",
    name: "nodeList",
    inputs: [],
    outputs: [{ type: "address[]" }],
    stateMutability: "view",
  },
  {
    type: "function",
    name: "nodeListAt",
    inputs: [{ type: "uint256", name: "index" }],
    outputs: [
      {
        type: "tuple",
        components: [
          { type: "address", name: "nodeAddress" },
          { type: "string", name: "url" },
          {
            type: "uint8",
            name: "status",
          },
          { type: "uint256", name: "amount" },
          { type: "uint256", name: "jobsCount" },
          { type: "string", name: "publicKey" },
        ],
      },
    ],
    stateMutability: "view",
  },
  {
    type: "function",
    name: "nodesCount",
    inputs: [],
    outputs: [{ type: "uint256" }],
    stateMutability: "view",
  },
  {
    type: "function",
    name: "activeNodesCount",
    inputs: [],
    outputs: [{ type: "uint256" }],
    stateMutability: "view",
  },
  {
    type: "function",
    name: "activeNodeList",
    inputs: [],
    outputs: [{ type: "address[]" }],
    stateMutability: "view",
  },
  {
    type: "function",
    name: "activeNodeListAt",
    inputs: [{ type: "uint256", name: "index" }],
    outputs: [
      {
        type: "tuple",
        components: [
          { type: "address", name: "nodeAddress" },
          { type: "string", name: "url" },
          {
            type: "uint8",
            name: "status",
          },
          { type: "uint256", name: "amount" },
          { type: "uint256", name: "jobsCount" },
          { type: "string", name: "publicKey" },
        ],
      },
    ],
    stateMutability: "view",
  },
  {
    type: "function",
    name: "getNode",
    inputs: [{ type: "address", name: "nodeAddress" }],
    outputs: [
      {
        type: "tuple",
        components: [
          { type: "address", name: "nodeAddress" },
          { type: "string", name: "url" },
          {
            type: "uint8",
            name: "status",
          },
          { type: "uint256", name: "amount" },
          { type: "uint256", name: "jobsCount" },
          { type: "string", name: "publicKey" },
        ],
      },
    ],
    stateMutability: "view",
  },
  {
    type: "function",
    name: "addNode",
    inputs: [
      { type: "address", name: "nodeAddress" },
      { type: "string", name: "url" },
      { type: "string", name: "publicKey" },
    ],
    outputs: [],
    stateMutability: "nonpayable",
  },
  {
    type: "function",
    name: "removeNode",
    inputs: [{ type: "address", name: "nodeAddress" }],
    outputs: [],
    stateMutability: "nonpayable",
  },
  {
    type: "function",
    name: "isNode",
    inputs: [{ type: "address", name: "nodeAddress" }],
    outputs: [{ type: "bool" }],
    stateMutability: "view",
  },
  {
    type: "function",
    name: "settlement",
    inputs: [],
    outputs: [{ type: "address" }],
    stateMutability: "view",
  },
  {
    type: "function",
    name: "updateSettlement",
    inputs: [{ type: "address", name: "newSettlement" }],
    outputs: [],
    stateMutability: "nonpayable",
  },
  {
    type: "function",
    name: "getAccount",
    inputs: [
      { type: "address", name: "user" },
      { type: "address", name: "node" },
    ],
    outputs: [
      {
        type: "tuple",
        components: [
          { type: "address", name: "user" },
          { type: "address", name: "node" },
          { type: "uint256", name: "nonce" },
          { type: "uint256", name: "balance" },
          { type: "uint256", name: "pendingRefund" },
          {
            type: "tuple[]",
            name: "refunds",
            components: [
              { type: "uint256", name: "index" },
              { type: "uint256", name: "amount" },
              { type: "uint256", name: "createdAt" },
              { type: "bool", name: "processed" },
            ],
          },
        ],
      },
    ],
    stateMutability: "view",
  },
  {
    type: "function",
    name: "getAccountPendingRefund",
    inputs: [
      { type: "address", name: "user" },
      { type: "address", name: "node" },
    ],
    outputs: [{ type: "uint256" }],
    stateMutability: "view",
  },
  {
    type: "function",
    name: "getAllAccounts",
    inputs: [],
    outputs: [
      {
        type: "tuple[]",
        components: [
          { type: "address", name: "user" },
          { type: "address", name: "node" },
          { type: "uint256", name: "nonce" },
          { type: "uint256", name: "balance" },
          { type: "uint256", name: "pendingRefund" },
          {
            type: "tuple[]",
            name: "refunds",
            components: [
              { type: "uint256", name: "index" },
              { type: "uint256", name: "amount" },
              { type: "uint256", name: "createdAt" },
              { type: "bool", name: "processed" },
            ],
          },
        ],
      },
    ],
    stateMutability: "view",
  },
  {
    type: "function",
    name: "accountExists",
    inputs: [
      { type: "address", name: "user" },
      { type: "address", name: "node" },
    ],
    outputs: [{ type: "bool" }],
    stateMutability: "view",
  },
  {
    type: "function",
    name: "addAccount",
    inputs: [
      { type: "address", name: "user" },
      { type: "address", name: "node" },
    ],
    outputs: [],
    stateMutability: "payable",
  },
  {
    type: "function",
    name: "deleteAccount",
    inputs: [
      { type: "address", name: "user" },
      { type: "address", name: "node" },
    ],
    outputs: [],
    stateMutability: "nonpayable",
  },
  {
    type: "function",
    name: "deposit",
    inputs: [
      { type: "address", name: "user" },
      { type: "address", name: "node" },
      { type: "uint256", name: "cancelRetrievingAmount" },
    ],
    outputs: [],
    stateMutability: "payable",
  },
  {
    type: "function",
    name: "request",
    inputs: [
      { type: "address", name: "user" },
      { type: "address", name: "node" },
    ],
    outputs: [],
    stateMutability: "nonpayable",
  },
  {
    type: "function",
    name: "process",
    inputs: [
      { type: "address", name: "user" },
      { type: "address", name: "node" },
    ],
    outputs: [
      { type: "uint256", name: "totalAmount" },
      { type: "uint256", name: "balance" },
      { type: "uint256", name: "pendingRefund" },
    ],
    stateMutability: "nonpayable",
  },
  {
    type: "function",
    name: "settlementFees",
    inputs: [
      {
        type: "tuple",
        name: "settlement",
        components: [
          { type: "bytes", name: "signature" },
          {
            type: "tuple",
            name: "data",
            components: [
              { type: "string", name: "id" },
              { type: "address", name: "user" },
              { type: "uint256", name: "cost" },
              { type: "uint256", name: "nonce" },
              { type: "bytes", name: "userSignature" },
            ],
          },
        ],
      },
    ],
    outputs: [],
    stateMutability: "nonpayable",
  },
];
export const SETTLEMENT_CONTRACT_ABI = [
  {
    type: "function",
    name: "version",
    inputs: [],
    outputs: [{ type: "uint256" }],
    stateMutability: "view",
  },
  {
    type: "function",
    name: "training",
    inputs: [],
    outputs: [{ type: "address" }],
    stateMutability: "view",
  },
  {
    type: "function",
    name: "updateTraining",
    inputs: [{ type: "address", name: "newTraining" }],
    outputs: [],
    stateMutability: "nonpayable",
  },
  {
    type: "function",
    name: "query",
    inputs: [],
    outputs: [{ type: "address" }],
    stateMutability: "view",
  },
  {
    type: "function",
    name: "updateQuery",
    inputs: [{ type: "address", name: "newQuery" }],
    outputs: [],
    stateMutability: "nonpayable",
  },
  {
    type: "function",
    name: "inference",
    inputs: [],
    outputs: [{ type: "address" }],
    stateMutability: "view",
  },
  {
    type: "function",
    name: "updateInference",
    inputs: [{ type: "address", name: "newInference" }],
    outputs: [],
    stateMutability: "nonpayable",
  },
  {
    type: "function",
    name: "pause",
    inputs: [],
    outputs: [],
    stateMutability: "nonpayable",
  },
  {
    type: "function",
    name: "unpause",
    inputs: [],
    outputs: [],
    stateMutability: "nonpayable",
  },
  {
    type: "function",
    name: "getUser",
    inputs: [{ type: "address", name: "user" }],
    outputs: [
      {
        type: "tuple",
        components: [
          { type: "address", name: "addr" },
          { type: "uint256", name: "availableBalance" },
          { type: "uint256", name: "totalBalance" },
          { type: "address[]", name: "queryNodes" },
          { type: "address[]", name: "inferenceNodes" },
          { type: "address[]", name: "trainingNodes" },
        ],
      },
    ],
    stateMutability: "view",
  },
  {
    type: "function",
    name: "getAllUsers",
    inputs: [],
    outputs: [
      {
        type: "tuple[]",
        components: [
          { type: "address", name: "addr" },
          { type: "uint256", name: "availableBalance" },
          { type: "uint256", name: "totalBalance" },
          { type: "address[]", name: "inferenceNodes" },
          { type: "address[]", name: "trainingNodes" },
        ],
      },
    ],
    stateMutability: "view",
  },
  {
    type: "function",
    name: "addUser",
    inputs: [],
    outputs: [],
    stateMutability: "payable",
  },
  {
    type: "function",
    name: "deleteUser",
    inputs: [],
    outputs: [],
    stateMutability: "nonpayable",
  },
  {
    type: "function",
    name: "deposit",
    inputs: [],
    outputs: [],
    stateMutability: "payable",
  },
  {
    type: "function",
    name: "withdraw",
    inputs: [{ type: "uint256", name: "amount" }],
    outputs: [],
    stateMutability: "nonpayable",
  },
  {
    type: "function",
    name: "depositTraining",
    inputs: [
      { type: "address", name: "node" },
      { type: "uint256", name: "amount" },
    ],
    outputs: [],
    stateMutability: "nonpayable",
  },
  {
    type: "function",
    name: "depositInference",
    inputs: [
      { type: "address", name: "node" },
      { type: "uint256", name: "amount" },
    ],
    outputs: [],
    stateMutability: "nonpayable",
  },
  {
    type: "function",
    name: "depositQuery",
    inputs: [
      { type: "address", name: "node" },
      { type: "uint256", name: "amount" },
    ],
    outputs: [],
    stateMutability: "nonpayable",
  },
  {
    type: "function",
    name: "retrieveTraining",
    inputs: [{ type: "address[]", name: "nodes" }],
    outputs: [],
    stateMutability: "nonpayable",
  },
  {
    type: "function",
    name: "retrieveInference",
    inputs: [{ type: "address[]", name: "nodes" }],
    outputs: [],
    stateMutability: "nonpayable",
  },
  {
    type: "function",
    name: "retrieveQuery",
    inputs: [{ type: "address[]", name: "nodes" }],
    outputs: [],
    stateMutability: "nonpayable",
  },
  {
    type: "function",
    name: "settlement",
    inputs: [
      { type: "address", name: "addr" },
      { type: "uint256", name: "cost" },
    ],
    outputs: [],
    stateMutability: "nonpayable",
  },
];

export class ContractConfig {
  dataRegistryAddress: string;
  verifiedComputingAddress: string;
  dataAnchoringTokenAddress: string;
  queryAddress: string;
  inferenceAddress: string;
  trainingAddress: string;
  settlementAddress: string;

  constructor(
    dataRegistryAddress: string = DEFAULT_DATA_REGISTRY_CONTRACT_ADDRESS,
    verifiedComputingAddress: string = DEFAULT_VERIFIED_COMPUTING_CONTRACT_ADDRESS,
    dataAnchoringTokenAddress: string = DEFAULT_DATA_ANCHORING_TOKEN_CONTRACT_ADDRESS,
    queryAddress: string = DEFAULT_QUERY_CONTRACT_ADDRESS,
    inferenceAddress: string = DEFAULT_INFERENCE_CONTRACT_ADDRESS,
    trainingAddress: string = DEFAULT_TRAINING_CONTRACT_ADDRESS,
    settlementAddress: string = DEFAULT_SETTLEMENT_CONTRACT_ADDRESS
  ) {
    this.dataRegistryAddress = dataRegistryAddress;
    this.verifiedComputingAddress = verifiedComputingAddress;
    this.dataAnchoringTokenAddress = dataAnchoringTokenAddress;
    this.queryAddress = queryAddress;
    this.inferenceAddress = inferenceAddress;
    this.trainingAddress = trainingAddress;
    this.settlementAddress = settlementAddress;
  }

  static local(): ContractConfig {
    return new ContractConfig(
      DEFAULT_DATA_REGISTRY_CONTRACT_ADDRESS,
      DEFAULT_VERIFIED_COMPUTING_CONTRACT_ADDRESS,
      DEFAULT_DATA_ANCHORING_TOKEN_CONTRACT_ADDRESS,
      DEFAULT_QUERY_CONTRACT_ADDRESS,
      DEFAULT_INFERENCE_CONTRACT_ADDRESS,
      DEFAULT_TRAINING_CONTRACT_ADDRESS,
      DEFAULT_SETTLEMENT_CONTRACT_ADDRESS
    );
  }

  static testnet(): ContractConfig {
    return new ContractConfig(
      TESTNET_DATA_REGISTRY_CONTRACT_ADDRESS,
      TESTNET_VERIFIED_COMPUTING_CONTRACT_ADDRESS,
      TESTNET_DATA_ANCHORING_TOKEN_CONTRACT_ADDRESS,
      TESTNET_QUERY_CONTRACT_ADDRESS,
      TESTNET_INFERENCE_CONTRACT_ADDRESS,
      TESTNET_TRAINING_CONTRACT_ADDRESS,
      TESTNET_SETTLEMENT_CONTRACT_ADDRESS
    );
  }
}
