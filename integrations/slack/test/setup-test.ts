/**
 * Simple test script to verify the bot setup
 */

import dotenv from 'dotenv';
import path from 'path';
import { config } from '../config/slack';
import { formatFileSize, validateFileType } from '../src/utils';

// Load environment variables from .env file
dotenv.config({ path: path.resolve(__dirname, '../.env') });

function runTests() {
    console.log('Running LazAI Slack Bot Tests...\n');

    // Test 1: Configuration
    console.log('Test 1: Configuration');
    console.log('Bot name:', config.bot.name);
    console.log('Max file size:', formatFileSize(config.bot.maxFileSize));
    console.log('Supported file types:', config.bot.supportedFileTypes.join(', '));
    console.log();

    // Test 2: Utility functions
    console.log('Test 2: Utility Functions');
    console.log('Format 1024 bytes:', formatFileSize(1024));
    console.log('Format 1MB:', formatFileSize(1024 * 1024));
    console.log('Validate JSON file:', validateFileType('data.json'));
    console.log('Validate invalid file:', validateFileType('script.exe'));
    console.log();

    // Test 3: Environment variables
    console.log('Test 3: Environment Variables');
    const requiredVars = [
        'SLACK_BOT_TOKEN',
        'SLACK_SIGNING_SECRET',
        'SLACK_APP_TOKEN',
        'PRIVATE_KEY',
        'IPFS_JWT'
    ];

    const missingVars = requiredVars.filter(varName => !process.env[varName]);
    
    if (missingVars.length === 0) {
        console.log('All environment variables are set');
    } else {
        console.log('Missing environment variables:', missingVars.join(', '));
        console.log('   Please check your .env file');
    }
    console.log();

    // Test 4: Port configuration
    console.log('Test 4: Server Configuration');
    console.log('Server port:', config.slack.port);
    console.log('Socket mode:', config.slack.socketMode ? 'Enabled' : 'Disabled');
    console.log();

    console.log('Test complete! Ready to start your LazAI Slack Bot');
    
    if (missingVars.length > 0) {
        console.log('\nFix the environment variables before running the bot');
        process.exit(1);
    }
}

// Run tests when script is executed directly
if (require.main === module) {
    runTests();
}

export { runTests };
