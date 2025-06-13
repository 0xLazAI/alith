import os
from typing import Optional

import aiohttp
from pydantic import BaseModel

from .interfaces import (
    DataStorage,
    FileMetadata,
    GetShareLinkOptions,
    StorageError,
    StorageType,
    UploadOptions,
)

IPFS_LINK = "https://ipfs.io"
IPFS_DWEB_LINK = "https://dweb.link"
IPFS_W3S_LINK = "https://w3s.link"
IPFS_TRUSTLESS_GATEWAY_LINK = "https://trustless-gateway.link"
IPFS_4EVERLAND_LINK = "https://4everland.io"
IPFS_PINATA_CLOUD_LINK = "https://gateway.pinata.cloud"
IPFS_NFT_STORAGE_LINK = "https://nftstorage.link"

IPFS_GATEWAY_ENV = "IPFS_GATEWAY"
IPFS_API_KEY_ENV = "IPFS_API_KEY"
IPFS_API_SECRET_ENV = "IPFS_API_SECRET_KEY"
IPFS_JWT_ENV = "IPFS_JWT"


class PinataUploadResponse(BaseModel):
    accept_duplicates: bool
    is_duplicate: Optional[bool] = None
    id: str
    user_id: str
    name: str
    cid: str
    size: int
    number_of_files: int
    mime_type: str
    group_id: Optional[str] = None
    created_at: str
    updated_at: str
    network: str
    streamable: bool
    vectorized: bool


class PinataIPFS(DataStorage):
    def __init__(self):
        self.client = aiohttp.ClientSession()
        self._configure_from_env()

    def _configure_from_env(self):
        self.gateway = os.getenv(IPFS_GATEWAY_ENV, IPFS_PINATA_CLOUD_LINK)
        self.api_key = os.getenv(IPFS_API_KEY_ENV)
        self.api_secret = os.getenv(IPFS_API_SECRET_ENV)
        self.jwt = os.getenv(IPFS_JWT_ENV)

    async def upload(self, opts: UploadOptions) -> FileMetadata:
        url = "https://uploads.pinata.cloud/v3/files"

        form = aiohttp.FormData()
        form.add_field("file", opts.data, filename=opts.name, content_type="text/plain")
        form.add_field("network", "public")

        headers = {"Authorization": f"Bearer {opts.token}"}

        try:
            async with self.client.post(url, data=form, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise StorageError(f"Pinata IPFS API error: {error_text}")

                data = await response.json()
                pinata_response = PinataUploadResponse(**data["data"])

                return FileMetadata(
                    id=pinata_response.cid,
                    name=pinata_response.name,
                    size=pinata_response.size,
                    modified_time=pinata_response.updated_at,
                )
        except aiohttp.ClientError as e:
            raise StorageError(f"Network error: {str(e)}") from e

    async def get_share_link(self, opts: GetShareLinkOptions) -> str:
        return f"https://gateway.pinata.cloud/ipfs/{opts.id}?download=true"

    def storage_type(self) -> StorageType:
        return StorageType.IPFS

    async def close(self):
        await self.client.close()
