use alloy::{
    primitives::{Address, address},
    sol,
};

sol! {
    event FileAdded(uint256 indexed fileId, address indexed ownerAddress, string url);
    event ProofAdded(uint256 indexed fileId, address indexed ownerAddress, uint256 proofIndex, string proofUrl);
    event PermissionGranted(uint256 indexed fileId, address indexed account);

    event RewardRequested(
        address indexed contributorAddress, uint256 indexed fileId, uint256 indexed proofIndex, uint256 rewardAmount
    );

    event PublicKeyUpdated(string newPublicKey);

    event TokenUpdated(address newToken);
    event VerifiedComputingUpdated(address newVerifiedComputing);

    error NotFileOwner();
    error FileUrlAlreadyUsed();
    error FileNotFound();
    error FileAlreadyRewarded();
    error NoPermission();
    error InvalidUrl();
    error InvalidAttestator(bytes32 messageHash, bytes signature, address signer);

    struct ProofData {
        uint256 id;
        string fileUrl;
        string proofUrl;
    }

    struct Proof {
        bytes signature;
        ProofData data;
    }

    struct Permission {
        address account;
        string key;
    }

    // `alloy::sol!`` does support solidity mapping here
    // struct File {
    //     uint256 id;
    //     address ownerAddress;
    //     string url;
    //     uint256 timestamp;
    //     uint256 proofIndex;
    //     uint256 proofsCount;
    //     uint256 rewardAmount;
    //     mapping(uint256 proofId => Proof proof) proofs;
    //     mapping(address account => string key) permissions;
    // }

    struct FileResponse {
        uint256 id;
        address ownerAddress;
        string url;
        uint256 proofIndex;
        uint256 rewardAmount;
    }

    // Data registry contract interface

    #[sol(rpc)]
    interface IDataRegistry {
        function name() external view returns (string memory);
        function version() external pure returns (uint256);
        function token() external view returns (DataAnchorToken);
        function verifiedComputing() external view returns (IVerifiedComputing);
        function updateVerifiedComputing(address newVerifiedComputing) external;

        // Public key operations

        function publicKey() external view returns (string memory);
        function updatePublicKey(string calldata newPublicKey) external;

        // Privacy data and file operations

        function addFile(string memory url) external returns (uint256);
        function addFileWithPermissions(string memory url, address ownerAddress, Permission[] memory permissions)
            external
            returns (uint256);
        function addPermissionForFile(uint256 fileId, address account, string memory key) external;

        // File view functions

        function getFile(uint256 fileId) external view returns (FileResponse memory);
        function getFileIdByUrl(string memory url) external view returns (uint256);
        function getFilePermission(uint256 fileId, address account) external view returns (string memory);
        function getFileProof(uint256 fileId, uint256 index) external view returns (Proof memory);
        function filesCount() external view returns (uint256);

        // Proof operations

        function addProof(uint256 fileId, Proof memory proof) external;

        // Request reward and token
        function requestReward(uint256 fileId, uint256 proofIndex) external;
    }

    event NodeAdded(address indexed nodeAddress);
    event NodeRemoved(address indexed nodeAddress);

    event JobSubmitted(uint256 indexed jobId, uint256 indexed fileId, address nodeAddress, uint256 bidAmount);
    event JobCanceled(uint256 indexed jobId);

    event JobComplete(address indexed attestator, uint256 indexed jobId, uint256 indexed fileId);
    event Claimed(address indexed nodeAddress, uint256 amount);

    error NodeAlreadyAdded();
    error NodeNotActive();
    error InvalidJobStatus();
    error InvalidJobNode();
    error NothingToClaim();
    error InsufficientFee();
    error NoActiveNode();
    error NotJobOwner();
    error TransferFailed();

    enum NodeStatus {
        None,
        Active,
        Removed
    }

    struct NodeInfo {
        address nodeAddress;
        string url;
        NodeStatus status;
        uint256 amount;
        uint256 jobsCount;
        string publicKey;
    }

    enum JobStatus {
        None,
        Submitted,
        Completed,
        Canceled
    }

    struct Job {
        uint256 fileId;
        uint256 bidAmount;
        JobStatus status;
        uint256 addedTimestamp;
        address ownerAddress;
        address nodeAddress;
    }

    // Verified Computing Contract for privacy data and inference in CPU/GPU TEE.

    #[sol(rpc)]
    interface IVerifiedComputing {
        function version() external pure returns (uint256);

        // Fee operations

        function nodeFee() external view returns (uint256);
        function updateNodeFee(uint256 newNodeFee) external;

        // Node operations

        function nodeList() external view returns (address[] memory);
        function nodeListAt(uint256 index) external view returns (NodeInfo memory);
        function nodesCount() external view returns (uint256);

        function activeNodesCount() external view returns (uint256);
        function activeNodeList() external view returns (address[] memory);
        function activeNodeListAt(uint256 index) external view returns (NodeInfo memory);

        function getNode(address nodeAddress) external view returns (NodeInfo memory);
        function addNode(address nodeAddress, string memory url, string memory publicKey) external;
        function removeNode(address nodeAddress) external;
        function isNode(address nodeAddress) external view returns (bool);

        function claim() external;

        function requestProof(uint256 fileId) external payable;

        function pause() external;
        function unpause() external;

        function submitJob(uint256 fileId) external payable;
        function completeJob(uint256 jobId) external;
        function fileJobIds(uint256 fileId) external view returns (uint256[] memory);
        function jobsCount() external view returns (uint256);
        function getJob(uint256 jobId) external view returns (Job memory);
    }

    // DAT Token

    interface IERC1155 {
        event TransferSingle(address indexed operator, address indexed from, address indexed to, uint256 id, uint256 value);
        event TransferBatch(
            address indexed operator,
            address indexed from,
            address indexed to,
            uint256[] ids,
            uint256[] values
        );
        event ApprovalForAll(address indexed account, address indexed operator, bool approved);
        event URI(string value, uint256 indexed id);

        function balanceOf(address account, uint256 id) external view returns (uint256);

        function balanceOfBatch(
            address[] calldata accounts,
            uint256[] calldata ids
        ) external view returns (uint256[] memory);

        function setApprovalForAll(address operator, bool approved) external;

        function isApprovedForAll(address account, address operator) external view returns (bool);

        function safeTransferFrom(address from, address to, uint256 id, uint256 value, bytes calldata data) external;

        function safeBatchTransferFrom(
            address from,
            address to,
            uint256[] calldata ids,
            uint256[] calldata values,
            bytes calldata data
        ) external;
    }

    #[sol(rpc)]
    contract DataAnchorToken is IERC1155 {
        function mint(address to, uint256 amount, string memory tokenURI_, bool verified_) public;
        function uri(uint256 tokenId) public view override returns (string memory);
        function verified(uint256 tokenId) public view returns (bool);
        function setTokenVerified(uint256 tokenId, bool verified_);
        function batchMint(address to, uint256[] memory ids, uint256[] memory amounts, string[] memory tokenURIs) public external;
    }
}

pub const DEFAULT_DATA_REGISTRY_CONTRACT_ADDRESS: Address =
    address!("0x969A0e040a9719a8FAd033a03dCA38542a0ef7DC");
pub const DEFAULT_DATA_VERIFIED_COMPUTING_CONTRACT_ADDRESS: Address =
    address!("0xEA30BA91F4DB33Ef0360Fc04d8E201954474dbD1");

#[derive(Debug, Clone)]
pub struct ContractConfig {
    pub data_registry_address: Address,
    pub verified_computing_address: Address,
}

impl Default for ContractConfig {
    fn default() -> Self {
        Self {
            data_registry_address: DEFAULT_DATA_REGISTRY_CONTRACT_ADDRESS,
            verified_computing_address: DEFAULT_DATA_VERIFIED_COMPUTING_CONTRACT_ADDRESS,
        }
    }
}
