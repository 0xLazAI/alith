# 🤖 LazAI Slack Bot

Easy data contribution to LazAI directly from Slack! This integration allows teams to contribute data files, check balances, and track contributions without leaving their workspace.

## ✨ Features

- 📤 **Contribute Files**: Upload and contribute data files directly from Slack
- 💰 **Check Balance**: View your DAT token balance and contribution stats
- 📊 **Track Status**: Monitor contribution verification status
- 🎯 **Progress Tracking**: Real-time progress updates during contribution
- 👥 **Team Integration**: Seamlessly integrates with team workflows

## 🚀 Quick Start

### For Users

1. Invite the bot to your channel: `/invite @LazAI`
2. Upload a data file to the channel
3. Type `/lazai-contribute` to contribute your file
4. Earn DAT tokens! 💰

### Available Commands

- `/lazai-help` - Show all available commands
- `/lazai-contribute` - Contribute uploaded files to LazAI
- `/lazai-balance` - Check your DAT token balance
- `/lazai-status` - View your contribution status and history

## 🛠️ Installation & Setup

### Prerequisites

- Node.js 16+ installed
- A Slack workspace with admin permissions
- LazAI wallet and IPFS credentials

### Step 1: Clone and Install

```bash
cd integrations/slack
npm install
```

### Step 2: Create Slack App

1. Go to https://api.slack.com/apps
2. Click "Create New App" → "From scratch"
3. Name it "LazAI Data Bot"
4. Choose your workspace

### Step 3: Configure Permissions

In your Slack app dashboard, go to "OAuth & Permissions" and add these Bot Token Scopes:

- `app_mentions:read`
- `chat:write`
- `commands`
- `files:read`
- `files:write`
- `users:read`

### Step 4: Enable Socket Mode

1. Go to "Socket Mode" in your app dashboard
2. Enable Socket Mode
3. Generate App-Level Token with `connections:write` scope
4. Copy the token (starts with `xapp-`)

### Step 5: Configure Environment

Copy your `.env` file with these values:

```env
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_SIGNING_SECRET=your-signing-secret-here
SLACK_APP_TOKEN=xapp-your-app-token-here
PRIVATE_KEY=your-wallet-private-key
IPFS_JWT=your-pinata-ipfs-jwt
PORT=3000
NODE_ENV=development
```

### Step 6: Add Slash Commands

In your Slack app dashboard, go to "Slash Commands" and create:

- `/lazai-help`
- `/lazai-contribute`
- `/lazai-balance`
- `/lazai-status`

### Step 7: Install to Workspace

1. Go to "Install App" in your app dashboard
2. Click "Install to Workspace"
3. Authorize the app

## 🏃 Running the Bot

### Development Mode

```bash
npm run dev
```

### Production Mode

```bash
npm run build
npm start
```

## 📖 Usage Examples

### Contributing a File

1. Upload a file (JSON, CSV, TXT, etc.) to any channel where the bot is present
2. Type `/lazai-contribute`
3. Select your file from the list
4. Watch the progress and earn DAT tokens!

### Checking Your Balance

```
/lazai-balance
```

Returns your current DAT balance and total contributions.

### Viewing Status

```
/lazai-status
```

Shows your recent contributions and their verification status.

## 🔧 Configuration

### Supported File Types

- `.txt` - Text files
- `.json` - JSON data
- `.csv` - Comma-separated values
- `.jsonl` - JSON Lines
- `.parquet` - Parquet files
- `.md` - Markdown files

### File Size Limits

- Maximum file size: 100MB
- Recommended size: 1-50MB for optimal processing

### Environment Variables

| Variable               | Description                     | Required |
| ---------------------- | ------------------------------- | -------- |
| `SLACK_BOT_TOKEN`      | Bot User OAuth Token            | ✅       |
| `SLACK_SIGNING_SECRET` | App signing secret              | ✅       |
| `SLACK_APP_TOKEN`      | App-level token for Socket Mode | ✅       |
| `PRIVATE_KEY`          | Your wallet private key         | ✅       |
| `IPFS_JWT`             | Pinata IPFS JWT token           | ✅       |
| `PORT`                 | Server port (default: 3000)     | ❌       |
| `NODE_ENV`             | Environment mode                | ❌       |

## 🧪 Development

### Project Structure

```
src/
├── index.ts       # Main application entry point
├── bot.ts         # Core bot logic and LazAI integration
├── commands.ts    # Slash command handlers
└── utils.ts       # Utility functions
```

### Adding New Commands

1. Add command handler in `src/commands.ts`
2. Register the command in your Slack app dashboard
3. Test in development mode

### Testing

```bash
# Run tests (when available)
npm test

# Check TypeScript compilation
npm run build

# Lint code
npm run lint
```

## 🤝 Contributing

### As a LazAI Dev Ambassador

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

### Contribution Ideas

- Add more file validation
- Implement batch file processing
- Add team analytics dashboard
- Create Slack workflow integrations
- Add notification preferences

## 🐛 Troubleshooting

### Common Issues

**Bot not responding to commands:**

- Check if Socket Mode is enabled
- Verify App-Level Token is correct
- Ensure bot is added to the channel

**File upload errors:**

- Check file size (max 100MB)
- Verify file type is supported
- Ensure IPFS credentials are valid

**Permission errors:**

- Verify bot token scopes
- Check if bot is properly installed to workspace
- Ensure user has permission to upload files

### Debug Mode

Set `NODE_ENV=development` and check console logs for detailed error messages.

## 📄 License

MIT License - see LICENSE file for details.

## 🙋 Support

- **Issues**: Open an issue on GitHub
- **Slack**: Join the LazAI community Slack
- **Docs**: Visit [LazAI Documentation](https://docs.lazai.com)
- **Discord**: LazAI Developer Community

## 🎉 Acknowledgments

Built with ❤️ by the LazAI Dev Ambassador community.

Special thanks to:

- Slack Bolt framework
- LazAI core team
- Contributing developers

---

**Ready to contribute data and earn DAT tokens? Get started now!** 🚀
