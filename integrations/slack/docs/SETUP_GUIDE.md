# 🚀 LazAI Slack Bot - Complete Setup Guide

This guide will walk you through setting up the LazAI Slack Bot step-by-step, perfect for beginners!

## 📋 Prerequisites Checklist

Before starting, make sure you have:

- [ ] **Node.js 16+** installed ([Download here](https://nodejs.org/))
- [ ] **Git** installed ([Download here](https://git-scm.com/))
- [ ] **Slack workspace** with admin permissions
- [ ] **LazAI wallet** with private key
- [ ] **Pinata account** for IPFS ([Sign up here](https://pinata.cloud/))

## 🔧 Step-by-Step Setup

### Step 1: Clone the Repository

```bash
# Navigate to your projects directory
cd c:\Users\Administrator\Desktop\alith\integrations\slack

# Install dependencies
npm install
```

### Step 2: Create Your Slack App

1. **Go to Slack API**: https://api.slack.com/apps
2. **Click "Create New App"**
3. **Choose "From scratch"**
4. **Fill in details**:
   - App Name: `LazAI Data Bot`
   - Workspace: Choose your workspace
5. **Click "Create App"**

### Step 3: Configure App Permissions

**In your new Slack app dashboard:**

1. **Go to "OAuth & Permissions"** (left sidebar)
2. **Scroll to "Bot Token Scopes"**
3. **Add these scopes** (click "Add an OAuth Scope"):
   ```
   app_mentions:read
   chat:write
   commands
   files:read
   files:write
   users:read
   ```

### Step 4: Enable Socket Mode

1. **Go to "Socket Mode"** (left sidebar)
2. **Toggle "Enable Socket Mode" to ON**
3. **Click "Generate Token"**
   - Token Name: `LazAI Connection`
   - Scope: `connections:write`
4. **Copy the generated token** (starts with `xapp-`)
5. **Click "Done"**

### Step 5: Install App to Workspace

1. **Go to "Install App"** (left sidebar)
2. **Click "Install to Workspace"**
3. **Click "Allow"** to authorize
4. **Copy the "Bot User OAuth Token"** (starts with `xoxb-`)

### Step 6: Get Your Signing Secret

1. **Go to "Basic Information"** (left sidebar)
2. **Scroll to "App Credentials"**
3. **Copy the "Signing Secret"**

### Step 7: Set Up Environment Variables

Create/update your `.env` file:

```env
# Slack Configuration (from steps above)
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_SIGNING_SECRET=your-signing-secret-here
SLACK_APP_TOKEN=xapp-your-app-token-here

# LazAI Configuration
PRIVATE_KEY=your-wallet-private-key
IPFS_JWT=your-pinata-ipfs-jwt

# Bot Configuration
PORT=3000
NODE_ENV=development
```

**How to get IPFS_JWT:**

1. Go to [Pinata.cloud](https://pinata.cloud/)
2. Sign up/login to your account
3. Go to API Keys section
4. Create a new API key with admin permissions
5. Copy the JWT token

### Step 8: Create Slash Commands

**In your Slack app dashboard:**

1. **Go to "Slash Commands"** (left sidebar)
2. **Click "Create New Command"** for each:

**Command 1: /lazai-help**

- Command: `/lazai-help`
- Request URL: `https://your-app.com/slack/events` (leave blank for Socket Mode)
- Short Description: `Show LazAI bot help`

**Command 2: /lazai-contribute**

- Command: `/lazai-contribute`
- Request URL: `https://your-app.com/slack/events` (leave blank for Socket Mode)
- Short Description: `Contribute data files to LazAI`

**Command 3: /lazai-balance**

- Command: `/lazai-balance`
- Request URL: `https://your-app.com/slack/events` (leave blank for Socket Mode)
- Short Description: `Check your DAT token balance`

**Command 4: /lazai-status**

- Command: `/lazai-status`
- Request URL: `https://your-app.com/slack/events` (leave blank for Socket Mode)
- Short Description: `View contribution status`

### Step 9: Test Your Setup

```bash
# Validate your configuration
npm run test:setup

# Expected output:
# ✅ All environment variables are set
# ✅ Server port: 3000
# 🎉 Test complete!
```

### Step 10: Start the Bot

```bash
# Development mode (with auto-restart)
npm run dev

# You should see:
# 🚀 LazAI Slack Bot is running!
# ⚡️ Bot started on port 3000
# 📱 Ready to receive Slack commands!
```

### Step 11: Test in Slack

1. **Go to your Slack workspace**
2. **Invite the bot to a channel**:
   ```
   /invite @LazAI Data Bot
   ```
3. **Test the help command**:
   ```
   /lazai-help
   ```
4. **You should see the help message!** 🎉

## ✅ Success Checklist

- [ ] Node.js and npm installed
- [ ] Repository cloned and dependencies installed
- [ ] Slack app created with proper name
- [ ] Bot token scopes configured
- [ ] Socket Mode enabled with app token
- [ ] App installed to workspace
- [ ] Environment variables configured
- [ ] Slash commands created
- [ ] Setup test passes (`npm run test:setup`)
- [ ] Bot starts successfully (`npm run dev`)
- [ ] Bot responds to `/lazai-help` in Slack

## 🐛 Common Issues & Solutions

### Issue: "Bot not responding"

**Solution**:

1. Check if Socket Mode is enabled
2. Verify your App Token starts with `xapp-`
3. Ensure bot is added to the channel

### Issue: "Permission denied errors"

**Solution**:

1. Verify all bot token scopes are added
2. Reinstall the app to workspace
3. Check if you have admin permissions

### Issue: "Environment variable errors"

**Solution**:

1. Double-check your `.env` file
2. Ensure no extra spaces in token values
3. Restart the bot after changes

### Issue: "IPFS upload fails"

**Solution**:

1. Verify your Pinata JWT token
2. Check if your Pinata account is active
3. Test with a small file first

## 🎯 Next Steps

Once your bot is working:

1. **Upload a test file** to your Slack channel
2. **Try `/lazai-contribute`** to contribute it
3. **Check your balance** with `/lazai-balance`
4. **View status** with `/lazai-status`
5. **Share with your team!** 🚀

## 💡 Pro Tips

- **Start with small files** (< 1MB) for testing
- **Use JSON or CSV files** for best compatibility
- **Test in a private channel** first before going public
- **Keep your environment variables secure**
- **Monitor the console logs** for debugging

## 🆘 Need Help?

If you get stuck:

1. Check the console logs for error messages
2. Run `npm run test:setup` to validate configuration
3. Refer to the troubleshooting section in README.md
4. Open an issue on GitHub with error details

**Happy contributing!** 🎉
