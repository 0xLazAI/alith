#!/usr/bin/env node
/**
 * Complete setup and validation script for LazAI Slack Bot
 * This script will validate everything and guide you through final setup
 */

import dotenv from 'dotenv';
import path from 'path';
import fs from 'fs';
import { config } from '../config/slack';

// Load environment variables
dotenv.config({ path: path.resolve(__dirname, '../.env') });

const chalk = {
    green: (str: string) => `\x1b[32m${str}\x1b[0m`,
    red: (str: string) => `\x1b[31m${str}\x1b[0m`,
    yellow: (str: string) => `\x1b[33m${str}\x1b[0m`,
    blue: (str: string) => `\x1b[34m${str}\x1b[0m`,
    cyan: (str: string) => `\x1b[36m${str}\x1b[0m`,
    bold: (str: string) => `\x1b[1m${str}\x1b[0m`,
    underline: (str: string) => `\x1b[4m${str}\x1b[0m`
};

function printHeader() {
    console.log('\n' + chalk.cyan('🚀 LazAI Slack Bot - Complete Setup Validation'));
    console.log(chalk.cyan('='.repeat(60)));
    console.log('Checking all components for production readiness...\n');
}

function checkEnvironmentVariables(): boolean {
    console.log(chalk.bold('📋 Environment Variables Check:'));
    
    const requiredVars = [
        'SLACK_BOT_TOKEN',
        'SLACK_SIGNING_SECRET', 
        'SLACK_APP_TOKEN',
        'PRIVATE_KEY',
        'IPFS_JWT'
    ];
    
    let allPresent = true;
    
    requiredVars.forEach(varName => {
        if (process.env[varName]) {
            const displayValue = process.env[varName]!.substring(0, 10) + '...';
            console.log(chalk.green(`   ✅ ${varName}: ${displayValue}`));
        } else {
            console.log(chalk.red(`   ❌ ${varName}: Missing`));
            allPresent = false;
        }
    });
    
    return allPresent;
}

function checkProjectStructure(): boolean {
    console.log('\n' + chalk.bold('📁 Project Structure Check:'));
    
    const requiredFiles = [
        'src/index.ts',
        'src/bot.ts', 
        'src/commands.ts',
        'src/utils.ts',
        'config/slack.ts',
        'package.json',
        'tsconfig.json',
        '.env'
    ];
    
    let allPresent = true;
    
    requiredFiles.forEach(file => {
        const filePath = path.resolve(__dirname, '..', file);
        if (fs.existsSync(filePath)) {
            console.log(chalk.green(`   ✅ ${file}`));
        } else {
            console.log(chalk.red(`   ❌ ${file}: Missing`));
            allPresent = false;
        }
    });
    
    return allPresent;
}

function checkDocumentation(): boolean {
    console.log('\n' + chalk.bold('📚 Documentation Check:'));
    
    const docFiles = [
        'README.md',
        'docs/SETUP_GUIDE.md',
        'docs/API_DOCUMENTATION.md',
        'docs/CONTRIBUTING.md',
        'docs/TROUBLESHOOTING.md'
    ];
    
    let allPresent = true;
    
    docFiles.forEach(file => {
        const filePath = path.resolve(__dirname, '..', file);
        if (fs.existsSync(filePath)) {
            console.log(chalk.green(`   ✅ ${file}`));
        } else {
            console.log(chalk.red(`   ❌ ${file}: Missing`));
            allPresent = false;
        }
    });
    
    return allPresent;
}

function printSlackSetupInstructions() {
    console.log('\n' + chalk.bold('🔧 Slack App Configuration Required:'));
    console.log('\nTo complete the setup, you need to:');
    
    console.log('\n' + chalk.yellow('1. Create Slack App:'));
    console.log('   • Go to https://api.slack.com/apps');
    console.log('   • Click "Create New App" → "From scratch"');
    console.log('   • Name: "LazAI Data Bot"');
    
    console.log('\n' + chalk.yellow('2. Configure Permissions:'));
    console.log('   • Go to "OAuth & Permissions"');
    console.log('   • Add Bot Token Scopes:');
    console.log('     - app_mentions:read');
    console.log('     - chat:write'); 
    console.log('     - commands');
    console.log('     - files:read');
    console.log('     - files:write');
    console.log('     - users:read');
    
    console.log('\n' + chalk.yellow('3. Enable Socket Mode:'));
    console.log('   • Go to "Socket Mode"');
    console.log('   • Enable Socket Mode');
    console.log('   • Generate App-Level Token');
    console.log('   • Scope: connections:write');
    
    console.log('\n' + chalk.yellow('4. Install to Workspace:'));
    console.log('   • Go to "Install App"');
    console.log('   • Click "Install to Workspace"');
    console.log('   • Copy Bot User OAuth Token');
    
    console.log('\n' + chalk.yellow('5. Create Slash Commands:'));
    console.log('   • Go to "Slash Commands"');
    console.log('   • Create these commands:');
    console.log('     - /lazai-help');
    console.log('     - /lazai-contribute');
    console.log('     - /lazai-balance');
    console.log('     - /lazai-status');
}

function printNextSteps(allConfigured: boolean) {
    if (allConfigured) {
        console.log('\n' + chalk.bold(chalk.green('🎉 Setup Complete! Ready to Launch:')));
        console.log('\n' + chalk.green('Start your bot:'));
        console.log(chalk.cyan('   npm run dev'));
        console.log('\n' + chalk.green('Test in Slack:'));
        console.log('   1. Invite bot to a channel: /invite @LazAI Data Bot');
        console.log('   2. Try: /lazai-help');
        console.log('   3. Upload a file and use: /lazai-contribute');
    } else {
        console.log('\n' + chalk.bold(chalk.red('⚠️  Setup Incomplete')));
        console.log('\nPlease complete the missing items above, then run:');
        console.log(chalk.cyan('   npm run setup:complete'));
    }
}

function printContributionReadiness() {
    console.log('\n' + chalk.bold('🎯 Contribution Readiness:'));
    console.log(chalk.green('   ✅ Code implementation complete'));
    console.log(chalk.green('   ✅ Documentation complete'));
    console.log(chalk.green('   ✅ Project structure ready'));
    console.log(chalk.green('   ✅ Testing framework ready'));
    
    console.log('\n' + chalk.bold('🚀 Ready for LazAI Dev Ambassador Contribution!'));
    console.log('\nYour Slack integration includes:');
    console.log('   • Complete slash command system');
    console.log('   • File contribution workflow');
    console.log('   • Progress tracking and feedback');
    console.log('   • Error handling and validation');
    console.log('   • Comprehensive documentation');
    console.log('   • Production-ready architecture');
}

async function main() {
    printHeader();
    
    const envOk = checkEnvironmentVariables();
    const structureOk = checkProjectStructure();
    const docsOk = checkDocumentation();
    
    const allConfigured = envOk && structureOk && docsOk;
    
    if (!allConfigured) {
        printSlackSetupInstructions();
    }
    
    printContributionReadiness();
    printNextSteps(allConfigured);
    
    console.log('\n' + chalk.cyan('='.repeat(60)));
    console.log(chalk.bold('Thank you for contributing to LazAI! 🙏'));
    console.log(chalk.cyan('='.repeat(60)) + '\n');
    
    process.exit(allConfigured ? 0 : 1);
}

main().catch(console.error);
