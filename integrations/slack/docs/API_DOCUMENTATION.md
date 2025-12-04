# 📚 LazAI Slack Bot API Documentation

This document provides detailed information about the bot's functionality, commands, and integration points.

## Bot Commands

### `/lazai-help`

**Description**: Displays help information and available commands.

**Usage**: `/lazai-help`

**Response**: Ephemeral message (only visible to user) with:

- List of available commands
- Getting started instructions
- Feature overview

**Example**:

```
User: /lazai-help
Bot: LazAI Bot Commands
     Available Commands:
     • /lazai-contribute - Contribute a data file
     • /lazai-balance - Check your DAT balance
     • /lazai-status - View contribution status
     ...
```

---

### `/lazai-contribute`

**Description**: Contribute recently uploaded files to the LazAI network.

**Usage**: `/lazai-contribute`

**Process**:

1. Scans last 20 messages in channel for files
2. Displays up to 10 most recent files
3. User selects file via button interaction
4. Bot processes and uploads to IPFS
5. Registers contribution on LazAI contract
6. Returns file ID and reward information

**Supported File Types**:

- `.txt` - Text files
- `.json` - JSON data
- `.csv` - Comma-separated values
- `.jsonl` - JSON Lines
- `.parquet` - Parquet files
- `.md` - Markdown files

**File Size Limits**:

- Minimum: 1 byte
- Maximum: 100 MB

**Response Flow**:

```
User: /lazai-contribute
Bot: Select File to Contribute
     [Shows list of recent files with "Contribute" buttons]

User: [Clicks "Contribute" on a file]
Bot: Contributing filename.json...
     [Progress bar: ████████░░ 80%]

Bot: Successfully contributed filename.json!
     🆔 File ID: FILE_1699123456789
     💰 Reward: 100 DAT
```

---

### `/lazai-balance`

**Description**: Check your current DAT token balance and contribution statistics.

**Usage**: `/lazai-balance`

**Response**: Ephemeral message with:

- Current DAT token balance
- Total number of files contributed
- Encouragement to contribute more

**Example**:

```
User: /lazai-balance
Bot: 💰 Your LazAI Balance

     DAT Tokens: 450 DAT
     Total Contributions: 12 files

     Keep contributing data to earn more DAT! 🚀
```

---

### `/lazai-status`

**Description**: View your recent contribution history and verification status.

**Usage**: `/lazai-status`

**Response**: Ephemeral message showing:

- Recent contributions (up to 10)
- File IDs and names
- Verification status (pending/verified/rewarded)
- Reward amounts
- Contribution dates

**Status Types**:

- 🔄 **Pending**: Contribution submitted, awaiting verification
- ✅ **Verified**: Verified by network nodes
- 💰 **Rewarded**: DAT tokens distributed

**Example**:

```
User: /lazai-status
Bot: 📊 Your Contribution Status

     Recent Contributions:
     💰 dataset_large.json
     🆔 ID: FILE_001 • 💰 150 DAT • ⏰ Nov 10, 2025

     ✅ training_data.csv
     🆔 ID: FILE_002 • 💰 100 DAT • ⏰ Nov 9, 2025
```

## 🔧 Bot Architecture

### Core Components

#### `SlackBot` Class

**File**: `src/bot.ts`

**Key Methods**:

- `contributeFile(fileUrl, fileName, userId)`: Main contribution logic
- `getUserBalance(userId)`: Retrieve user's DAT balance
- `getContributionStatus(userId)`: Get recent contribution history
- `downloadSlackFile(fileUrl)`: Download files from Slack
- `createProgressBlock(step, totalSteps, message)`: Generate progress UI

#### Command Handlers

**File**: `src/commands.ts`

**Structure**:

- Each command has dedicated handler function
- Handlers use Slack Bolt framework patterns
- Interactive components (buttons) handled separately
- Error handling with user-friendly messages

#### Utility Functions

**File**: `src/utils.ts`

**Functions**:

- `formatFileSize(bytes)`: Human-readable file sizes
- `validateFileType(filename)`: Check supported formats
- `generateProgressBar(current, total)`: Progress visualization

## 🔄 Contribution Workflow

### 1. File Discovery

```typescript
// Get recent messages from channel
const result = await client.conversations.history({
  channel: command.channel_id,
  limit: 20,
});

// Extract files from messages
const filesInChannel = result.messages
  ?.filter((msg: any) => msg.files && msg.files.length > 0)
  .flatMap((msg: any) => msg.files || [])
  .filter((file: any) => file && file.url_private)
  .slice(0, 10);
```

### 2. File Validation

```typescript
// Size validation
if (fileData.length > 100 * 1024 * 1024) {
  return { success: false, error: "File too large" };
}

// Type validation (implicit through UI)
const supportedTypes = [".txt", ".json", ".csv", ".jsonl", ".parquet", ".md"];
```

### 3. File Processing

```typescript
// Download from Slack
const fileData = await downloadSlackFile(fileUrl);

// Encrypt data
const password = wallet.sign(encryptionSeed).signature;
const encryptedData = encrypt(fileData, password);

// Upload to IPFS
const fileMeta = await ipfs.upload({
  name: fileName,
  data: encryptedData,
  token: process.env.IPFS_JWT,
});
```

### 4. Blockchain Registration

```typescript
// Get shareable URL
const url = await ipfs.getShareLink({ token, id: fileMeta.id });

// Register on LazAI contract
let fileId = await client.getFileIdByUrl(url);
if (fileId === BigInt(0)) {
  fileId = await client.addFile(url);
}

// Request verification and reward
await client.requestProof(fileId, reward);
await client.requestReward(fileId);
```

## 📊 Data Structures

### ContributionResult

```typescript
interface ContributionResult {
  success: boolean;
  fileId?: string; // Unique identifier for contributed file
  reward?: number; // DAT tokens earned
  error?: string; // Error message if failed
  message: string; // User-friendly status message
}
```

### SlackFile

```typescript
interface SlackFile {
  id: string; // Slack's internal file ID
  name: string; // Original filename
  url_private: string; // Private download URL
  size?: number; // File size in bytes
  filetype?: string; // File extension
  mimetype?: string; // MIME type
}
```

### UserBalance

```typescript
interface UserBalance {
  balance: number; // Current DAT token balance
  contributions: number; // Total files contributed
}
```

### ContributionStatus

```typescript
interface ContributionStatus {
  recent: Array<{
    fileId: string; // LazAI file identifier
    fileName: string; // Original filename
    status: "pending" | "verified" | "rewarded";
    reward: number; // DAT tokens for this file
    date: string; // Contribution timestamp
  }>;
}
```

## ⚙️ Configuration

### Environment Variables

| Variable               | Type   | Required | Description                                           |
| ---------------------- | ------ | -------- | ----------------------------------------------------- |
| `SLACK_BOT_TOKEN`      | string | ✅       | Bot User OAuth Token (starts with `xoxb-`)            |
| `SLACK_SIGNING_SECRET` | string | ✅       | App signing secret for verification                   |
| `SLACK_APP_TOKEN`      | string | ✅       | App-level token for Socket Mode (starts with `xapp-`) |
| `PRIVATE_KEY`          | string | ✅       | Wallet private key for signing transactions           |
| `IPFS_JWT`             | string | ✅       | Pinata JWT token for IPFS uploads                     |
| `PORT`                 | number | ❌       | Server port (default: 3000)                           |
| `NODE_ENV`             | string | ❌       | Environment mode (`development`/`production`)         |

### Bot Configuration

```typescript
const config = {
  bot: {
    maxFileSize: 100 * 1024 * 1024, // 100MB
    supportedFileTypes: [".txt", ".json", ".csv", ".jsonl", ".parquet", ".md"],
    defaultReward: 100, // DAT tokens
  },
  slack: {
    maxFilesPerCommand: 10, // Files shown in /lazai-contribute
    historyLimit: 20, // Messages scanned for files
  },
};
```

## 🔒 Security Considerations

### Data Encryption

- Files are encrypted using wallet-signed keys
- Encryption seed: `"Sign to retrieve your encryption key"`
- Uses symmetric encryption (AES) with RSA key wrapping

### Access Control

- Bot only accesses public channel files
- Users can only contribute files they uploaded
- Private channels require explicit bot invitation

### Token Security

- Environment variables should never be committed
- Bot tokens have minimum required permissions
- App tokens are scoped to `connections:write` only

## 🧪 Testing

### Manual Testing Checklist

1. **Setup Validation**:

   ```bash
   npm run test:setup
   ```

2. **Command Testing**:

   - [ ] `/lazai-help` shows help message
   - [ ] `/lazai-contribute` lists recent files
   - [ ] File contribution completes successfully
   - [ ] `/lazai-balance` shows balance
   - [ ] `/lazai-status` shows contribution history

3. **File Type Testing**:

   - [ ] JSON files (.json)
   - [ ] CSV files (.csv)
   - [ ] Text files (.txt)
   - [ ] Large files (near 100MB limit)
   - [ ] Invalid file types (should be filtered)

4. **Error Scenarios**:
   - [ ] No files in channel
   - [ ] Files too large
   - [ ] Network/IPFS errors
   - [ ] Invalid tokens/permissions

### Automated Testing

```bash
# Run all tests
npm test

# Validate setup
npm run test:setup

# Build validation
npm run validate
```

## 🚀 Deployment

### Development

```bash
npm run dev
```

### Production

```bash
npm run build
npm start
```

### Docker (Optional)

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
CMD ["npm", "start"]
```

## 📈 Monitoring & Analytics

### Metrics to Track

- Total contributions per day/week
- Average file sizes
- User engagement (active contributors)
- Success/failure rates
- Response times

### Logging

- All contributions logged with timestamps
- Error tracking with stack traces
- Performance metrics for file processing
- User activity patterns

## 🔄 Updates & Maintenance

### Regular Maintenance

- Monitor Slack API changes
- Update dependencies monthly
- Review error logs weekly
- Performance optimization quarterly

### Feature Roadmap

- Batch file contribution
- Team analytics dashboard
- Custom reward settings
- Integration with Slack Workflows
- Mobile app notifications

---

**For technical support, consult the troubleshooting guide or open an issue on GitHub.**
