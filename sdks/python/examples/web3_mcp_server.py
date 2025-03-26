from mcp.server.fastmcp import FastMCP
from web3 import Web3

# Initialize the MCP server
mcp = FastMCP("MetisBlockServer")

# Connect to Metis mainnet
w3 = Web3(Web3.HTTPProvider("https://andromeda.metis.io/?owner=1088"))


# MCP Tool: Fetch Latest Block Number (already working)
@mcp.tool()
def get_latest_block() -> int:
    """Fetches the latest block number from Metis mainnet."""
    return w3.eth.block_number


# MCP Tool: Fetch Previous Block Number (already working)
@mcp.tool()
def get_previous_block() -> int:
    """Fetches the previous block number from Metis mainnet."""
    latest_block = get_latest_block()
    return latest_block - 1


# Run the MCP server
if __name__ == "__main__":
    mcp.run()
