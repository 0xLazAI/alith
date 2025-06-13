import asyncio
from os import getenv

from alith.data.storage import (
    GetShareLinkOptions,
    PinataIPFS,
    StorageError,
    UploadOptions,
)


async def main():
    ipfs = PinataIPFS()
    try:
        data = "Your data"
        name = "your_file.txt"
        token = getenv("IPFS_JWT", "")

        file_meta = await ipfs.upload(
            UploadOptions(name=name, data=data.encode(), token=token)
        )
        print(f"Upload file to the Pinata IPFS: {file_meta}")
        print(
            f"Share link: {await ipfs.get_share_link(GetShareLinkOptions(token=token, id=file_meta.id))}"
        )
    except StorageError as e:
        print(f"Error: {e}")
    finally:
        await ipfs.close()


if __name__ == "__main__":
    asyncio.run(main())
