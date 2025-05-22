use alloy::{
    primitives::{Address, address},
    sol,
};

sol! {
    event FileAdded(
        uint256 indexed fileId,
        address indexed ownerAddress,
        string url
    );

    event PermissionGranted(
        uint256 indexed fileId,
        address indexed account
    );

    event ProofAdded(
        uint256 indexed fileId,
        address indexed ownerAddress,
        uint256 proofIndex,
        string proofUrl
    );

    event PublicKeyUpdated(string newPublicKey);

    struct ProofData {
        uint256 id;
        string fileUrl;
        string proofUrl;
        string instruction;
    }

    struct Proof {
        bytes signature;
        ProofData data;
    }

    struct Permission {
        address account;
        string key;
    }

    struct File {
        uint256 id;
        address ownerAddress;
        string url;
        uint256 modifiedTimestamp;
        bool verified;
    }

    // Data registry contract interface

    #[sol(rpc)]
    interface IDataRegistry {
        // Role View functions

        function ADMIN_ROLE() view returns (bytes32);
        function MAINTAINER_ROLE() view returns (bytes32);

        // Public key operations

        function publicKey() external view returns (string memory);
        function updatePublicKey(string calldata newPublicKey) external;

        // Privacy data and file operations

        function addFile(string memory url) returns (uint256);
        function addFileWithPermissions(
            string memory url,
            address ownerAddress,
            Permission[] memory permissions
        ) returns (uint256);
        function addPermissionForFile(uint256 fileId, address account, string memory key);

        // File view functions

        function getFile(uint256 fileId) view returns (File memory);
        function getFileIdByUrl(string memory url) view returns (uint256);
        function getFilePermission(uint256 fileId, address account) view returns (string memory);
        function getFileProof(uint256 fileId, uint256 index) view returns (Proof memory);
        function getFilesCount() view returns (uint256);

        // Proof operations

        function addProof(uint256 fileId, Proof memory proof);

        // Role operations

        function getRoleAdmin(bytes32 role) view returns (bytes32);
        function grantRole(bytes32 role, address account);
        function hasRole(bytes32 role, address account) view returns (bool);
        function renounceRole(bytes32 role, address callerConfirmation);
        function revokeRole(bytes32 role, address account);
        function setRoleAdmin(bytes32 role, bytes32 adminRole);

        // Owner and Contract addresses

        function tokenAddress() external view returns (DataAnchorToken);
        function verifiedComputingAddress() external view returns (IVerifiedComputing);

        // Request reward and token
        function requestReward(uint256 fileId, uint256 proofIndex) external;
    }

    event NodeAdded(address indexed nodeAddress);

    event NodeRemoved(address indexed nodeAddress);

    struct NodeInfo {
        address nodeAddress;
        string url;
        uint8 status;
        uint256 amount;
        string publicKey;
    }

    // Verified Computing Contract for privacy data and inference in CPU/GPU TEE.

    #[sol(rpc)]
    interface IVerifiedComputing {
        // Role View functions

        function ADMIN_ROLE() view returns (bytes32);
        function MAINTAINER_ROLE() view returns (bytes32);

        // Data registry contract functions

        function dataRegistryAddress() view returns (address);
        function updateDataRegistryAddress(address newDataRegistryAddress);

        // Fee operations

        function fee() view returns (uint256);
        function updateFee(uint256 newFee);

        // Node operations

        function nodeList() view returns (address[] memory);
        function nodeListAt(uint256 index) view returns (NodeInfo memory);
        function nodeCount() view returns (uint256);

        function getNode(address nodeAddress) view returns (NodeInfo memory);
        function addNode(address nodeAddress, string memory url, string memory publicKey);
        function removeNode(address nodeAddress);

        function claim();

        function requestProof(uint256 fileId) external payable;
    }

    // DAT Token

    #[sol(rpc)]
    interface IERC721 {
        event Transfer(address indexed from, address indexed to, uint256 indexed tokenId);
        event Approval(address indexed owner, address indexed approved, uint256 indexed tokenId);
        event ApprovalForAll(address indexed owner, address indexed operator, bool approved);
        function balanceOf(address owner) external view returns (uint256 balance);
        function ownerOf(uint256 tokenId) external view returns (address owner);
        function safeTransferFrom(address from, address to, uint256 tokenId, bytes calldata data) external;
        function safeTransferFrom(address from, address to, uint256 tokenId) external;
        function transferFrom(address from, address to, uint256 tokenId) external;
        function approve(address to, uint256 tokenId) external;
        function setApprovalForAll(address operator, bool approved) external;
        function getApproved(uint256 tokenId) external view returns (address operator);
        function isApprovedForAll(address owner, address operator) external view returns (bool);
    }

    #[sol(rpc)]
    contract DataAnchorToken is IERC721 {

    }
}

pub const DEFAULT_DATA_REGISTRY_CONTRACT_ADDRESS: Address =
    address!("0x4141410000000000000000000000000000000000");
pub const DEFAULT_DATA_VERIFIED_COMPUTING_CONTRACT_ADDRESS: Address =
    address!("0x4242420000000000000000000000000000000000");

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
