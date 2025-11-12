# Troubleshooting Guide - LazAI Slack Bot

This guide helps you diagnose and fix common issues with the LazAI Slack Bot.

## 🔍 Quick Diagnosis

### Run the Setup Test

```bash
npm run test:setup
```

This will check:

- Environment variables are set
- Configuration is valid
- Dependencies are installed
- Ports are available

## Bot Not Starting

### Issue: "Missing environment variables"

**Symptoms**:

```
Missing environment variables: SLACK_BOT_TOKEN, SLACK_APP_TOKEN
```

**Solution**:

1. Check your `.env` file exists
2. Verify all required variables are set:
   ```env
   SLACK_BOT_TOKEN=xoxb-...
   SLACK_SIGNING_SECRET=...
   SLACK_APP_TOKEN=xapp-...
   PRIVATE_KEY=...
   IPFS_JWT=...
   ```
3. Ensure no extra spaces or quotes around values
4. Restart the bot: `npm run dev`

### Issue: "Port already in use"

**Symptoms**:

```
Error: listen EADDRINUSE :::3000
```

**Solution**:

```bash
# Option 1: Change port in .env
PORT=3001

# Option 2: Kill process using port 3000
# Windows:
netstat -ano | findstr :3000
taskkill /PID <PID_NUMBER> /F

# macOS/Linux:
lsof -ti:3000 | xargs kill -9
```

### Issue: "Socket connection failed"

**Symptoms**:

```
Error: WebSocket connection failed
```

**Solution**:

1. Verify Socket Mode is enabled in Slack app
2. Check App Token starts with `xapp-`
3. Ensure App Token has `connections:write` scope
4. Regenerate App Token if needed

## 🔌 Slack Integration Issues

### Issue: "Bot not responding to commands"

**Symptoms**: Commands like `/lazai-help` show "Command not found"

**Solution**:

1. **Check if bot is added to channel**:

   ```
   /invite @LazAI Data Bot
   ```

2. **Verify commands are created in Slack app**:

   - Go to Slack API dashboard
   - Navigate to "Slash Commands"
   - Ensure all commands exist:
     - `/lazai-help`
     - `/lazai-contribute`
     - `/lazai-balance`
     - `/lazai-status`

3. **Check bot permissions**:

   - Go to "OAuth & Permissions"
   - Verify these scopes are added:
     - `app_mentions:read`
     - `chat:write`
     - `commands`
     - `files:read`
     - `files:write`
     - `users:read`

4. **Reinstall bot to workspace**:
   - Go to "Install App"
   - Click "Reinstall to Workspace"

### Issue: "Permission denied errors"

**Symptoms**:

```
Error: missing_scope
Error: not_in_channel
```

**Solution**:

1. Add missing bot token scopes
2. Invite bot to private channels manually
3. Ensure workspace admin approval
4. Check if bot token is valid (not expired)

### Issue: "Commands show but don't respond"

**Symptoms**: Commands appear in Slack but no response from bot

**Solution**:

1. **Check bot is running**:

   ```bash
   npm run dev
   # Should show: "LazAI Slack Bot is running!"
   ```

2. **Verify Socket Mode configuration**:

   - Socket Mode must be enabled
   - App Token must be valid
   - Bot must be connected (check console logs)

3. **Check console for errors**:
   - Look for connection errors
   - Verify authentication success
   - Check for rate limiting issues

## File Upload Issues

### Issue: "No files found in recent messages"

**Symptoms**: `/lazai-contribute` shows "No files found"

**Solution**:

1. **Upload a test file to the channel**:

   - Drag and drop a small JSON/CSV file
   - Wait for upload to complete
   - Try `/lazai-contribute` again

2. **Check file upload timing**:

   - Bot scans last 20 messages
   - Files must be uploaded recently
   - Upload file and immediately run command

3. **Verify file permissions**:
   - Ensure files are visible to bot
   - Check if files are in private threads
   - Try uploading to public channel

### Issue: "File too large" errors

**Symptoms**:

```
File is too large (max 100MB)
```

**Solution**:

1. **Check actual file size**:

   ```bash
   ls -lh your-file.json
   ```

2. **Compress large files**:

   ```bash
   # Zip the file
   zip compressed-data.zip large-file.csv

   # Or split into smaller files
   split -b 50M large-file.csv chunk_
   ```

3. **Use supported formats**:
   - Prefer compressed formats (`.parquet` over `.csv`)
   - Remove unnecessary whitespace from JSON
   - Use `.jsonl` instead of large JSON arrays

### Issue: "IPFS upload failures"

**Symptoms**:

```
Error: IPFS upload failed: 401 Unauthorized
Error: Request timeout
```

**Solution**:

1. **Verify Pinata credentials**:

   ```bash
   # Test your JWT token
   curl -X GET \
     -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     https://api.pinata.cloud/data/testAuthentication
   ```

2. **Check Pinata account limits**:

   - Log into Pinata dashboard
   - Verify storage quota
   - Check if account is active

3. **Network troubleshooting**:

   ```bash
   # Test connectivity
   ping gateway.pinata.cloud

   # Check firewall/proxy settings
   # Ensure ports 80/443 are open
   ```

## Balance & Status Issues

### Issue: "Balance shows 0 DAT"

**Symptoms**: `/lazai-balance` always shows 0 tokens

**Current behavior**: The bot uses mock data for demonstration

**Solution**:

1. **This is expected behavior** - the bot currently simulates LazAI integration
2. **For real integration**, you would need to:
   - Connect to actual LazAI smart contracts
   - Implement proper wallet integration
   - Add real blockchain transaction handling

### Issue: "Status shows no contributions"

**Symptoms**: `/lazai-status` shows empty history

**Solution**:

1. **Contribute a test file first**:

   - Upload a small file to channel
   - Use `/lazai-contribute` to contribute it
   - Wait for processing to complete
   - Check `/lazai-status` again

2. **For development**: The bot uses mock historical data

## Development Issues

### Issue: "TypeScript compilation errors"

**Symptoms**:

```
error TS2304: Cannot find name 'SlackCommandMiddlewareArgs'
```

**Solution**:

```bash
# Clean and rebuild
npm run clean
npm install
npm run build

# Check TypeScript version
npx tsc --version
```

### Issue: "Module import errors"

**Symptoms**:

```
Cannot find module '@slack/bolt'
Error: Cannot resolve dependency
```

**Solution**:

```bash
# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install

# Or use specific versions
npm install @slack/bolt@^3.14.0
```

### Issue: "Hot reload not working"

**Symptoms**: Code changes don't trigger restart

**Solution**:

```bash
# Use nodemon directly
npx nodemon --exec ts-node src/index.ts

# Or check nodemon config
cat nodemon.json

# Ensure file watching is enabled
npm run dev -- --watch
```

## 🌐 Network & Connectivity

### Issue: "WebSocket connection errors"

**Symptoms**:

```
WebSocket connection to 'wss://...' failed
Connection timeout
```

**Solution**:

1. **Check internet connection**:

   ```bash
   ping slack.com
   nslookup wss-primary.slack.com
   ```

2. **Firewall/Proxy issues**:

   - Ensure WebSocket connections are allowed
   - Check corporate firewall settings
   - Try different network connection

3. **VPN conflicts**:
   - Disable VPN temporarily
   - Use different VPN server
   - Configure VPN to allow WebSocket traffic

### Issue: "Rate limiting errors"

**Symptoms**:

```
Error: rate_limited
Too many requests
```

**Solution**:

1. **Reduce request frequency**:

   - Add delays between operations
   - Implement exponential backoff
   - Cache responses when possible

2. **Check Slack app tier limits**:
   - Free tier: 1 request per second
   - Paid tier: Higher limits
   - Consider upgrading workspace plan

## Advanced Debugging

### Enable Debug Logging

```bash
# Set debug environment
NODE_ENV=development npm run dev

# Or add debug logging to code
console.log('Debug info:', { fileId, userId, timestamp });
```

### Check System Resources

```bash
# Memory usage
node --inspect src/index.ts

# CPU usage
top -p $(pgrep -f "node.*slack")

# Disk space
df -h
```

### Network Debugging

```bash
# Monitor network requests
npm install -g mitmproxy
mitmweb

# Check DNS resolution
nslookup api.slack.com
dig slack.com

# Test SSL certificates
openssl s_client -connect api.slack.com:443
```

## 📋 Diagnostic Checklist

When reporting issues, include:

### Environment Information

- [ ] Operating system and version
- [ ] Node.js version (`node --version`)
- [ ] NPM version (`npm --version`)
- [ ] Bot version/commit hash

### Configuration

- [ ] `.env` file contents (hide sensitive tokens)
- [ ] `package.json` dependencies
- [ ] Slack app configuration screenshots

### Error Details

- [ ] Complete error messages
- [ ] Console logs (last 50 lines)
- [ ] Steps to reproduce
- [ ] Expected vs actual behavior

### Testing Results

- [ ] `npm run test:setup` output
- [ ] Manual command testing results
- [ ] Network connectivity tests

## 🆘 Getting Help

### Self-Help Resources

1. **Documentation**: Read API docs and setup guide
2. **GitHub Issues**: Search existing issues
3. **Stack Overflow**: Search Slack Bolt framework issues
4. **Slack API Docs**: Official Slack documentation

### Community Support

- **GitHub Discussions**: Ask questions and share solutions
- **Discord**: Real-time help from community
- **Office Hours**: Weekly community calls (check calendar)

### Escalation Process

1. **GitHub Issue**: Create detailed bug report
2. **Tag maintainers**: Use appropriate labels
3. **Emergency**: Contact via email for critical issues

### Response Times

- **Community**: Best effort, usually 24-48 hours
- **Maintainers**: 24-72 hours for bug reports
- **Critical issues**: Within 24 hours

## Prevention Tips

### Regular Maintenance

```bash
# Weekly dependency updates
npm update

# Monthly security audit
npm audit
npm audit fix

# Periodic cleanup
npm run clean
npm install
```

### Best Practices

- **Monitor logs** regularly for warnings
- **Test in development** before deploying
- **Keep tokens secure** and rotate periodically
- **Backup configuration** files
- **Document customizations** you make

### Health Monitoring

```bash
# Create health check endpoint
npm run start -- --health-check

# Monitor uptime
curl http://localhost:3000/health

# Set up alerts for failures
```

---

**Still having issues? Don't hesitate to open a GitHub issue with detailed information!**
