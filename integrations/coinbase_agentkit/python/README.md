# Coinbase Agentkit Alith Python Integration

This integration provides the following methods:

1. **Use Agentkit actions within Alith**: You can directly leverage a variety of actions from the agentkit ecosystem without the need to rewrite them in Alith.

```python
import json
import os

from coinbase_agentkit import (
    AgentKit,
    AgentKitConfig,
    CdpWalletProvider,
    CdpWalletProviderConfig,
    cdp_api_action_provider,
    cdp_wallet_action_provider,
    erc20_action_provider,
    pyth_action_provider,
    wallet_action_provider,
    weth_action_provider,
)
from coinbase_agentkit_alith import get_alith_tools
from alith import Agent
from dotenv import load_dotenv

# Configure a file to persist the agent's CDP API Wallet Data.
wallet_data_file = "wallet_data.txt"

load_dotenv()


def initialize_agent() -> Agent:
    """Initialize the agent with CDP Agentkit."""

    # Initialize CDP Wallet Provider
    wallet_data = None
    if os.path.exists(wallet_data_file):
        with open(wallet_data_file) as f:
            wallet_data = f.read()

    cdp_config = None
    if wallet_data is not None:
        cdp_config = CdpWalletProviderConfig(wallet_data=wallet_data)

    wallet_provider = CdpWalletProvider(cdp_config)

    agentkit = AgentKit(
        AgentKitConfig(
            wallet_provider=wallet_provider,
            action_providers=[
                cdp_api_action_provider(),
                cdp_wallet_action_provider(),
                erc20_action_provider(),
                pyth_action_provider(),
                wallet_action_provider(),
                weth_action_provider(),
            ],
        )
    )

    wallet_data_json = json.dumps(wallet_provider.export_wallet().to_dict())

    with open(wallet_data_file, "w") as f:
        f.write(wallet_data_json)

    # use get_langchain_tools
    tools = get_alith_tools(agentkit)

    preamble = (
        "Be creative and do something interesting on the blockchain. "
        "Choose an action or set of actions and execute it that highlights your abilities."
    )

    # Create an Alith Agent and CDP Agentkit tools.
    return Agent(
        name="A dummy Agent",
        model="gpt-4o-mini",
        preamble=preamble,
        tools=tools,
    )


agent = initialize_agent()
print(agent.prompt("Transfer 0.5 ETH to 0xAABB"))
```

## Reference

- [Agentkit GitHub](https://github.com/coinbase/agentkit)
