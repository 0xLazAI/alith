import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


class StorageType(enum.Enum):
    GOOGLE_DRIVE = "google-drive"
    DROPBOX = "dropbox"
    IPFS = "ipfs"


class StorageError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@dataclass
class UploadOptions:
    name: str
    data: bytes
    token: str


@dataclass
class GetShareLinkOptions:
    token: str
    id: str


@dataclass
class FileMetadata:
    id: str
    name: str
    size: int
    modified_time: Optional[str] = None


class DataStorage(ABC):
    @abstractmethod
    async def upload(self, opts: "UploadOptions") -> "FileMetadata":
        pass

    @abstractmethod
    async def get_share_link(self, opts: "GetShareLinkOptions") -> str:
        pass

    @abstractmethod
    def storage_type(self) -> StorageType:
        pass
