export class StorageError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "StorageError";
  }
}

export interface UploadOptions {
  name: string;
  data: Buffer;
  token: string;
}

export interface GetShareLinkOptions {
  token: string;
  id: string;
}

export interface FileMetadata {
  id: string;
  name: string;
  size: number;
  modifiedTime?: string | null;
}

export enum StorageType {
  GoogleDrive = "google-drive",
  Dropbox = "dropbox",
  IPFS = "ipfs",
}

export interface DataStorage {
  upload(opts: UploadOptions): Promise<FileMetadata>;
  getShareLink(opts: GetShareLinkOptions): Promise<string>;
  storageType(): StorageType;
}
