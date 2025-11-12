import { App } from '@slack/bolt';
import { SlackBot } from './bot';
import { formatFileSize } from './utils';

// Type definitions for Slack Bolt
type SlashCommandArgs = {
    command: any;
    ack: () => Promise<void>;
    respond: (message: any) => Promise<void>;
    client?: any;
};
      
type ActionArgs = {
    body: any;
    ack: () => Promise<void>;
    respond: (message: any) => Promise<void>;
};

export function setupCommands(app: App, bot: SlackBot) {
    
    // /lazai-help - Show all available commands
    app.command('/lazai-help', async ({ command, ack, respond }: SlashCommandArgs) => {
        await ack();
        
        const helpBlocks = [
            {
                type: 'header',
                text: {
                    type: 'plain_text',
                    text: '🤖 LazAI Bot Commands'
                }
            },
            {
                type: 'section',
                text: {
                    type: 'mrkdwn',
                    text: '*Available Commands:*'
                }
            },
            {
                type: 'section',
                text: {
                    type: 'mrkdwn',
                    text: '• `/lazai-contribute` - Contribute a data file\n' +
                          '• `/lazai-balance` - Check your DAT balance\n' +
                          '• `/lazai-status` - View contribution status\n' +
                          '• `/lazai-analytics` - Get contribution analytics\n' +
                          '• `/lazai-help` - Show this help message'
                }
            },
            {
                type: 'section',
                text: {
                    type: 'mrkdwn',
                    text: '*Getting Started:*\n' +
                          '1. Upload a file to this channel\n' +
                          '2. Use `/lazai-contribute` to contribute it\n' +
                          '3. Earn DAT tokens for your data!'
                }
            }
        ];

        await respond({
            text: 'LazAI Bot Help',
            blocks: helpBlocks,
            response_type: 'ephemeral'
        });
    });

    // /lazai-contribute - Contribute data files
    app.command('/lazai-contribute', async ({ command, ack, respond, client }: SlashCommandArgs) => {
        await ack();
        
        try {
            // Get recent files from the channel
            const result = await client.conversations.history({
                channel: command.channel_id,
                limit: 20
            });

            const filesInChannel = result.messages
                ?.filter((msg: any) => msg.files && msg.files.length > 0)
                .flatMap((msg: any) => msg.files || [])
                .filter((file: any) => file && file.url_private)
                .slice(0, 10); // Latest 10 files

            if (!filesInChannel || filesInChannel.length === 0) {
                await respond({
                    text: '📂 No files found in recent messages. Please upload a file first!',
                    response_type: 'ephemeral'
                });
                return;
            }

            // Create file selection blocks
            const fileBlocks = [
                {
                    type: 'header',
                    text: {
                        type: 'plain_text',
                        text: '📂 Select File to Contribute'
                    }
                },
                {
                    type: 'section',
                    text: {
                        type: 'mrkdwn',
                        text: 'Choose a file from recent uploads:'
                    }
                }
            ];

            // Add file options
            filesInChannel.forEach((file: any, index: number) => {
                if (file && file.name && file.url_private) {
                    fileBlocks.push({
                        type: 'section',
                        text: {
                            type: 'mrkdwn',
                            text: `📄 *${file.name}*\n💾 Size: ${formatFileSize(file.size || 0)}\n📅 Type: ${file.filetype || 'unknown'}`
                        },
                        accessory: {
                            type: 'button',
                            text: {
                                type: 'plain_text',
                                text: 'Contribute'
                            },
                            action_id: 'contribute_file',
                            value: JSON.stringify({
                                url: file.url_private,
                                name: file.name,
                                size: file.size
                            })
                        }
                    } as any);
                }
            });

            await respond({
                text: 'Select a file to contribute',
                blocks: fileBlocks,
                response_type: 'ephemeral'
            });

        } catch (error) {
            console.error('Error in contribute command:', error);
            await respond({
                text: 'Error retrieving files. Please try again.',
                response_type: 'ephemeral'
            });
        }
    });

    // Handle file contribution button clicks
    app.action('contribute_file', async ({ body, ack, respond }: ActionArgs) => {
        await ack();
        
        try {
            const fileInfo = JSON.parse((body as any).actions[0].value);
            const userId = (body as any).user.id;
            
            // Show progress message
            await respond({
                text: `🔄 Contributing ${fileInfo.name}...`,
                blocks: bot.createProgressBlock(1, 4, `Processing ${fileInfo.name}`),
                response_type: 'ephemeral'
            });

            // Contribute the file
            const result = await bot.contributeFile(fileInfo.url, fileInfo.name, userId);
            
            if (result.success) {
                await respond({
                    text: result.message,
                    blocks: [
                        {
                            type: 'section',
                            text: {
                                type: 'mrkdwn',
                                text: result.message
                            }
                        },
                        {
                            type: 'context',
                            elements: [
                                {
                                    type: 'mrkdwn',
                                    text: `Contributed by <@${userId}> • ${new Date().toLocaleString()}`
                                }
                            ]
                        }
                    ],
                    response_type: 'in_channel' // Show success to everyone
                });
            } else {
                await respond({
                    text: result.message,
                    response_type: 'ephemeral'
                });
            }

        } catch (error) {
            console.error('Error contributing file:', error);
            await respond({
                text: 'Error contributing file. Please try again.',
                response_type: 'ephemeral'
            });
        }
    });

    // /lazai-balance - Check DAT balance
    app.command('/lazai-balance', async ({ command, ack, respond }: SlashCommandArgs) => {
        await ack();
        
        try {
            const userId = command.user_id;
            const balance = await bot.getUserBalance(userId);
            
            const balanceBlocks = [
                {
                    type: 'header',
                    text: {
                        type: 'plain_text',
                        text: '💰 Your LazAI Balance'
                    }
                },
                {
                    type: 'section',
                    fields: [
                        {
                            type: 'mrkdwn',
                            text: `*DAT Tokens:*\n${balance.balance} DAT`
                        },
                        {
                            type: 'mrkdwn',
                            text: `*Total Contributions:*\n${balance.contributions} files`
                        }
                    ]
                },
                {
                    type: 'context',
                    elements: [
                        {
                            type: 'mrkdwn',
                            text: `Keep contributing data to earn more DAT!`
                        }
                    ]
                }
            ];

            await respond({
                text: `Balance: ${balance.balance} DAT`,
                blocks: balanceBlocks,
                response_type: 'ephemeral'
            });

        } catch (error) {
            console.error('Error getting balance:', error);
            await respond({
                text: 'Error retrieving balance. Please try again.',
                response_type: 'ephemeral'
            });
        }
    });

    // /lazai-status - Check contribution status
    app.command('/lazai-status', async ({ command, ack, respond }: SlashCommandArgs) => {
        await ack();
        
        try {
            const userId = command.user_id;
            const status = await bot.getContributionStatus(userId);
            
            const statusBlocks = [
                {
                    type: 'header',
                    text: {
                        type: 'plain_text',
                        text: '📊 Your Contribution Status'
                    }
                }
            ];

            if (status.recent.length === 0) {
                statusBlocks.push({
                    type: 'section',
                    text: {
                        type: 'mrkdwn',
                        text: '📂 No recent contributions found.\nStart contributing data to earn DAT tokens!'
                    }
                });
            } else {
                statusBlocks.push({
                    type: 'section',
                    text: {
                        type: 'mrkdwn',
                        text: '*Recent Contributions:*'
                    }
                });

                status.recent.forEach(contrib => {
                const statusEmoji = {
                    pending: '[PENDING]',
                    verified: '[VERIFIED]', 
                    rewarded: '[REWARDED]'
                }[contrib.status];                    statusBlocks.push({
                        type: 'section',
                        text: {
                            type: 'mrkdwn',
                            text: `${statusEmoji} *${contrib.fileName}*\n` +
                                  `🆔 ID: ${contrib.fileId} • 💰 ${contrib.reward} DAT • ⏰ ${new Date(contrib.date).toLocaleDateString()}`
                        }
                    });
                });
            }

            await respond({
                text: 'Your contribution status',
                blocks: statusBlocks,
                response_type: 'ephemeral'
            });

        } catch (error) {
            console.error('Error getting status:', error);
            await respond({
                text: 'Error retrieving status. Please try again.',
                response_type: 'ephemeral'
            });
        }
    });
}


