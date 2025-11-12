import { App } from '@slack/bolt';
import dotenv from 'dotenv';
import { SlackBot } from './bot';
import { setupCommands } from './commands';

// Load environment variables
dotenv.config();

async function main() {
    try {
        // Initialize Slack app
        const app = new App({
            token: process.env.SLACK_BOT_TOKEN,
            signingSecret: process.env.SLACK_SIGNING_SECRET,
            socketMode: true,
            appToken: process.env.SLACK_APP_TOKEN,
        });

        // Initialize LazAI bot
        const lazaiBot = new SlackBot(app);
        
        // Setup all slash commands
        setupCommands(app, lazaiBot);

        // Start the app
        const port = process.env.PORT || 3000;
        await app.start(port);
        
        console.log('LazAI Slack Bot is running!');
        console.log(`Bot started on port ${port}`);
        console.log('Ready to receive Slack commands!');
        
    } catch (error) {
        console.error('Error starting bot:', error);
        process.exit(1);
    }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
    console.log('\n👋 Shutting down LazAI Slack Bot...');
    process.exit(0);
});

main();
