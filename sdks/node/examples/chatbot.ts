import { Agent } from "alith";
import * as readline from "readline";

const agent = new Agent({
    model: "your model name", // your preferred model
    apiKey: "your api key", // use your api key
    baseUrl: "base url"
});

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

console.log("Alith Chatbot is ready!");
console.log("Ask your question:");
console.log("Type 'exit' to quit.");

function ask() {
    rl.question("You: ", async (user_input) => {
        if (user_input.toLowerCase() === "exit") {
            rl.close();
            return;
        }
        const response = await agent.prompt(user_input);
        console.log("Bot:", response);
        ask();
    });
}

ask();
