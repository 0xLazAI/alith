import { App } from '@slack/bolt';
import { WebClient } from '@slack/web-api';
import axios from 'axios';
import FormData from 'form-data';

export interface ContributionResult {
    success: boolean;
    fileId?: string;
    reward?: number;
    error?: string;
    message: string;
}

export class SlackBot {
    private app: App;
    private client: WebClient;

    constructor(app: App) {
        this.app = app;
        this.client = app.client;
    }

    async contributeFile(fileUrl: string, fileName: string, userId: string): Promise<ContributionResult> {
        try {
            console.log(`📤 Contributing file: ${fileName} for user: ${userId}`);

            // Download file from Slack
            const fileData = await this.downloadSlackFile(fileUrl);
            if (!fileData) {
                return {
                    success: false,
                    error: 'Failed to download file',
                    message: 'Could not download the file from Slack'
                };
            }

            // Validate file size (max 100MB)
            if (fileData.length > 100 * 1024 * 1024) {
                return {
                    success: false,
                    error: 'File too large',
                    message: 'File is too large (max 100MB)'
                };
            }

            // Here we would integrate with the LazAI client
            // For now, simulate the contribution process
            const result = await this.simulateContribution(fileData, fileName);
            
            return {
                success: true,
                fileId: result.fileId,
                reward: result.reward,
                message: `Successfully contributed ${fileName}!\nFile ID: ${result.fileId}\nReward: ${result.reward} DAT`
            };

        } catch (error) {
            console.error('Error contributing file:', error);
            return {
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error',
                message: '❌ Failed to contribute file. Please try again.'
            };
        }
    }

    private async downloadSlackFile(fileUrl: string): Promise<Buffer | null> {
        try {
            const response = await axios.get(fileUrl, {
                headers: {
                    'Authorization': `Bearer ${process.env.SLACK_BOT_TOKEN}`
                },
                responseType: 'arraybuffer'
            });
            
            return Buffer.from(response.data);
        } catch (error) {
            console.error('Error downloading file:', error);
            return null;
        }
    }

    private async simulateContribution(fileData: Buffer, fileName: string) {
        // Simulate contribution process (replace with actual LazAI integration)
        await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate processing time
        
        return {
            fileId: `FILE_${Date.now()}`,
            reward: 100,
            url: `https://ipfs.io/ipfs/mock-hash-${Date.now()}`
        };
    }

    async getUserBalance(userId: string): Promise<{ balance: number; contributions: number }> {
        // Simulate getting user balance (replace with actual LazAI integration)
        return {
            balance: Math.floor(Math.random() * 1000) + 100,
            contributions: Math.floor(Math.random() * 50) + 5
        };
    }

    async getContributionStatus(userId: string): Promise<{
        recent: Array<{
            fileId: string;
            fileName: string;
            status: 'pending' | 'verified' | 'rewarded';
            reward: number;
            date: string;
        }>;
    }> {
        // Simulate recent contributions (replace with actual LazAI integration)
        const mockContributions = [
            {
                fileId: 'FILE_001',
                fileName: 'dataset_1.json',
                status: 'rewarded' as const,
                reward: 100,
                date: new Date(Date.now() - 86400000).toISOString() // 1 day ago
            },
            {
                fileId: 'FILE_002', 
                fileName: 'training_data.csv',
                status: 'verified' as const,
                reward: 150,
                date: new Date(Date.now() - 172800000).toISOString() // 2 days ago
            }
        ];

        return { recent: mockContributions };
    }

    async sendMessage(channel: string, text: string, blocks?: any[]) {
        try {
            await this.client.chat.postMessage({
                channel,
                text,
                blocks
            });
        } catch (error) {
            console.error('Error sending message:', error);
        }
    }

    async sendEphemeralMessage(channel: string, user: string, text: string, blocks?: any[]) {
        try {
            await this.client.chat.postEphemeral({
                channel,
                user,
                text,
                blocks
            });
        } catch (error) {
            console.error('Error sending ephemeral message:', error);
        }
    }

    createProgressBlock(step: number, totalSteps: number, message: string) {
        const progress = Math.round((step / totalSteps) * 100);
        const progressBar = '█'.repeat(Math.round(progress / 10)) + '░'.repeat(10 - Math.round(progress / 10));
        
        return [
            {
                type: 'section',
                text: {
                    type: 'mrkdwn',
                    text: `*${message}*\n\`${progressBar}\` ${progress}%`
                }
            }
        ];
    }
}
