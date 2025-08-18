import {
  type DataStorage,
  type FileMetadata,
  type GetShareLinkOptions,
  StorageError,
  StorageType,
  type UploadOptions,
} from "./interfaces";

export const IPFS_LINK: string = "https://ipfs.io";
export const IPFS_DWEB_LINK: string = "https://dweb.link";
export const IPFS_W3S_LINK: string = "https://w3s.link";
export const IPFS_TRUSTLESS_GATEWAY_LINK: string =
  "https://trustless-gateway.link";
export const IPFS_4EVERLAND_LINK: string = "https://4everland.io";
export const IPFS_PINATA_CLOUD_LINK: string = "https://gateway.pinata.cloud";
export const IPFS_NFT_STORAGE_LINK: string = "https://nftstorage.link";

export const IPFS_GATEWAY_ENV: string = "IPFS_GATEWAY";
export const IPFS_API_KEY_ENV: string = "IPFS_API_KEY";
export const IPFS_API_SECRET_ENV: string = "IPFS_API_SECRET_KEY";
export const IPFS_JWT_ENV: string = "IPFS_JWT";

export class PinataFileDetails {
  acceptDuplicates: boolean;
  isDuplicate: boolean | null;
  id: string;
  userId: string | null;
  name: string;
  cid: string;
  size: number;
  numberOfFiles: number;
  mimeType: string;
  groupId: string | null;
  createdAt: string;
  updatedAt: string;
  network: string;
  streamable: boolean;
  vectorized: boolean | null;

  constructor() {
    this.acceptDuplicates = false;
    this.isDuplicate = false;
    this.id = "";
    this.userId = "";
    this.name = "";
    this.cid = "";
    this.size = 0;
    this.numberOfFiles = 0;
    this.mimeType = "";
    this.groupId = null;
    this.createdAt = "";
    this.updatedAt = "";
    this.network = "";
    this.streamable = false;
    this.vectorized = false;
  }
}

export class PinataUploadResponse {
  data: PinataFileDetails;

  constructor() {
    this.data = new PinataFileDetails();
  }
}

export class PinataIPFS implements DataStorage {
  static new(): PinataIPFS {
    return new PinataIPFS();
  }

  async upload(opts: UploadOptions): Promise<FileMetadata> {
    const { name, data, token } = opts;
    const url = "https://uploads.pinata.cloud/v3/files";

    const formData = new FormData();
    const filePart = new Blob([data.toString()], { type: "text/plain" });
    formData.append("file", filePart, name);
    formData.append("network", "public");

    try {
      const response = await fetch(url, {
        method: "POST",
        body: formData,
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new StorageError(`Pinata IPFS API error: ${errorText}`);
      }

      const resp = (await response.json()) as PinataUploadResponse;
      return PinataFileDetailsToMetadata(resp.data);
    } catch (error) {
      throw new StorageError(`Upload error: ${error}`);
    }
  }

  async getShareLink(opts: GetShareLinkOptions): Promise<string> {
    return `https://gateway.pinata.cloud/ipfs/${opts.id}?download=true`;
  }

  storageType(): StorageType {
    return StorageType.IPFS;
  }
}

function PinataFileDetailsToMetadata(
  pinataFileDetails: PinataFileDetails
): FileMetadata {
  return {
    id: pinataFileDetails.cid,
    name: pinataFileDetails.name,
    size: pinataFileDetails.size,
    modifiedTime: pinataFileDetails.updatedAt || null,
  };
}
