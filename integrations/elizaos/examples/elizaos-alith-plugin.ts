import { createAlithPlugin, Agent } from "elizaos-alith";
import {
	AgentRuntime,
	CacheManager,
	MemoryCacheAdapter,
	ModelProviderName,
} from "@elizaos/core";

const provider = ModelProviderName.OPENAI;
const agent = new Agent({
	name: "A dummy Agent",
	model: "gpt-4",
	preamble:
		"You are a comedian here to entertain the user using humour and jokes.",
	provider,
});
const runtime = new AgentRuntime({
	token: "your-token",
	databaseAdapter: new YourDatabaseAdapter(),
	cacheManager: new CacheManager(new MemoryCacheAdapter()),
	modelProvider: provider,
	// Add Alith plugin in the ElizaOS agent runtime.
	plugins: [
		createAlithPlugin(agent),
		// Omit other plugins
	],
});
